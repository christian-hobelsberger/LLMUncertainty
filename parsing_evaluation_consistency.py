import re
import json
import ast
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List

# =====================
# UNIVERSAL ROBUST PARSER
# =====================
def universal_answer_parser(output: str) -> Dict[str, Any]:
    """
    Universal parser that handles various answer formats robustly.
    
    Handles patterns like:
    - {"answer": "text"}}
    - Answer: text
    - Multiple JSON objects in sequence
    - Malformed JSON with missing brackets
    
    Returns: {"answer": str|None}
    """
    if not output or pd.isna(output):
        return {"answer": None}
    
    output = str(output).strip()
    if not output:
        return {"answer": None}
    
    # Strategy 1: Try to find valid JSON objects (including multiple)
    json_matches = re.findall(r'\{[^{}]*\}', output, re.DOTALL)
    for json_match in json_matches:
        try:
            parsed = json.loads(json_match)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = parsed.get("answer", "").strip()
                if answer:  # Only return if we have a non-empty answer
                    return {"answer": answer}
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Strategy 2: Try to fix malformed JSON (missing brackets, trailing characters)
    # Look for patterns like: "answer": "text"}
    json_pattern = r'"answer"\s*:\s*"([^"]+)"'
    json_match = re.search(json_pattern, output, re.IGNORECASE | re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()
        if answer:
            return {"answer": answer}
    
    # Strategy 3: Look for Answer:
    answer_patterns = [
        r'Answer\s*[:=]?\s*([^\n\r]+?))',
        r'answer\s*[:=]?\s*([^\n\r]+?)'
    ]
    
    answer = None
    
    # Extract answer
    for pattern in answer_patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            candidate_answer = match.group(1).strip()
            # Clean up common artifacts
            candidate_answer = re.sub(r'^[.,:;}\]]+|[.,:;{\[]+$', '', candidate_answer).strip()
            candidate_answer = re.sub(r'^\s*[-•*]\s*', '', candidate_answer).strip()
            if candidate_answer and len(candidate_answer) > 0:
                answer = candidate_answer
                break
    
    # Strategy 4: If no structured format found, try to extract from first meaningful sentence
    if not answer:
        # Remove common prefixes and get first substantial text
        cleaned = re.sub(r'^(Answer|answer)\s*[:=]?\s*', '', output, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?\n]', cleaned)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2 and not sentence.lower().startswith('confidence'):
                # Clean up artifacts
                sentence = re.sub(r'[{}"\[\]]+', '', sentence).strip()
                if sentence:
                    answer = sentence
                    break
    
    # Strategy 5: Last resort - take first substantial word/phrase
    if not answer:
        words = re.findall(r'\b\w+(?:\s+\w+)*\b', output)
        if words:
            answer = words[0]
    
    return {"answer": answer}

def normalize_text_answer(answer: str) -> str:
    """Normalize text answers: lowercase, remove punctuation, normalize whitespace."""
    if not answer:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r'[.,!?;:\"\'\(\)\[\]]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer

# =====================
# TRIVIAQA PARSING & EVAL
# =====================
# normalize_trivia_answer replaced by normalize_text_answer

def parse_trivia_output(output: str) -> dict:
    """
    Parses TriviaQA model output using universal parser, then normalizes for text answers.
    Returns dict with key 'answer'.
    """
    parsed = universal_answer_parser(output)
    answer = normalize_text_answer(parsed.get("answer")) if parsed.get("answer") else None
    return {"answer": answer}


def evaluate_trivia_answer(predicted: str, gold_answers: list, threshold: float = 0.8) -> bool:
    """
    Compares a predicted answer to a list of gold answers with fuzzy and substring matching.
    """
    if not predicted or not gold_answers:
        return False
    predicted = normalize_text_answer(predicted)
    for gold in gold_answers:
        gold_norm = normalize_text_answer(str(gold))
        if predicted == gold_norm:
            return True
        if len(predicted) > 2 and len(gold_norm) > 2:
            if SequenceMatcher(None, predicted, gold_norm).ratio() >= threshold:
                return True
        if len(predicted) >= 3 and (predicted in gold_norm or gold_norm in predicted):
            return True
    return False


def parse_and_evaluate_trivia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses TriviaQA outputs, evaluates correctness using flexible gold answer extraction.
    Expects:
    - df["model_output"]
    - df["answers"] as a list-like column (actual list or stringified list)
    Adds: parsed_answer, parsed_confidence, is_correct
    """
    parsed = df["model_output"].apply(parse_trivia_output)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    def safe_parse_answers(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                return [val]
        return [val] if val else []
    def evaluate_row(row):
        pred = row["parsed_answer"]
        golds = safe_parse_answers(row.get("answers"))
        return evaluate_trivia_answer(pred, golds)
    df["is_correct"] = df.apply(evaluate_row, axis=1)
    return df


# =====================
# SQUAD PARSING & EVAL
# =====================
# normalize_squad_answer replaced by normalize_text_answer

def parse_squad_output(output: str) -> dict:
    """
    Parses SQuAD model output using universal parser, then normalizes for text answers.
    Returns dict with key 'answer'.
    """
    parsed = universal_answer_parser(output)
    answer = normalize_text_answer(parsed.get("answer")) if parsed.get("answer") else None
    return {"answer": answer}


def evaluate_squad_answer(predicted: str, gold_answers: list, threshold: float = 0.8) -> bool:
    """
    Compares a predicted answer to a list of gold answers with fuzzy and substring matching.
    Similar to TriviaQA evaluation but handles SQuAD-specific cases including unanswerable questions.
    """
    if not predicted or not gold_answers:
        return False
    
    predicted = normalize_text_answer(predicted)
    
    # Handle unanswerable questions - check if predicted answer indicates "no answer"
    no_answer_indicators = {"", "no answer", "unanswerable", "cannot answer", "not answerable", "no", "none"}
    if predicted in no_answer_indicators:
        # Check if any gold answer is also empty/unanswerable
        return any(normalize_text_answer(str(gold)) in no_answer_indicators for gold in gold_answers)
    
    for gold in gold_answers:
        gold_norm = normalize_text_answer(str(gold))
        
        # Exact match
        if predicted == gold_norm:
            return True
        
        # Fuzzy matching for longer answers
        if len(predicted) > 2 and len(gold_norm) > 2:
            if SequenceMatcher(None, predicted, gold_norm).ratio() >= threshold:
                return True
        
        # Substring matching
        if len(predicted) >= 3 and (predicted in gold_norm or gold_norm in predicted):
            return True
    
    return False

def parse_and_evaluate_squad(df: pd.DataFrame, remove_unanswerable: bool = False) -> pd.DataFrame:
    """
    Parses SQuAD outputs, evaluates correctness using flexible gold answer extraction.
    
    Args:
        df: DataFrame with columns:
            - model_output: The model's raw output
            - answers: List of acceptable answers (can be stringified list)
            - is_impossible (optional): Boolean indicating if question is unanswerable
        remove_unanswerable: If True, removes unanswerable questions from the dataset
    
    Returns:
        DataFrame with added columns: parsed_answer, parsed_confidence, is_correct
    """
    # Parse model outputs
    parsed = df["model_output"].apply(parse_squad_output)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    
    def safe_parse_answers(val):
        """Converts stringified lists or values into real lists."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                return [val]
        return [val] if val else []
    
    def evaluate_row(row):
        pred = row["parsed_answer"]
        golds = safe_parse_answers(row.get("answers"))
        return evaluate_squad_answer(pred, golds)
    
    df["is_correct"] = df.apply(evaluate_row, axis=1)
    
    # Remove unanswerable questions if requested
    if remove_unanswerable and "is_impossible" in df.columns:
        df_filtered = df[~df["is_impossible"]].copy()
        print(f"Removed {(df['is_impossible'] == True).sum()} unanswerable questions. {len(df_filtered)} questions remaining.")
        return df_filtered
    
    return df

    
# =====================
# MULTI-SAMPLE TRIVIAQA PARSING & AGGREGATION
# =====================
def normalize_trivia_multi(ans):
    """
    Normalize TriviaQA answers for multi-sample aggregation.
    Returns normalized lowercase string for consistency.
    """
    if pd.isna(ans):
        return None
    ans = str(ans).strip().lower()
    ans = re.sub(r'[.,!?;:\"\'\(\)\[\]]', '', ans)
    ans = re.sub(r'\s+', ' ', ans).strip()
    return ans if ans else None

def parse_and_evaluate_trivia_multi(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """
    Parse TriviaQA outputs for multi-sample data and add parsed columns.
    Expects columns: model_output, answers, and a grouping column (default: question_id)
    Adds: parsed_answer
    """
    # Parse only if not already present
    if "parsed_answer" not in df.columns:
        parsed = df["model_output"].apply(parse_trivia_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    
    return df

def aggregate_trivia(df: pd.DataFrame, group_col: str = "question_id", 
                    similarity_threshold: float = 0.8) -> pd.DataFrame:
    """
    xxx
    """
    def agg_group(group):
        # Only consider rows with valid answers
        group_valid = group.dropna(subset=["parsed_answer"]).copy()
        group_valid["norm_answer"] = group_valid["parsed_answer"].apply(normalize_trivia_multi)
        group_valid = group_valid.dropna(subset=["norm_answer"])

        if group_valid.empty:
            return pd.Series({
                "agg_answer": None,
                "answer_variants": 0
            })

        # Cluster similar answers together
        answer_clusters = {}
        for _, row in group_valid.iterrows():
            answer = row["norm_answer"]
            
            # Find if this answer is similar to any existing cluster
            found_cluster = False
            for cluster_key in answer_clusters.keys():
                if SequenceMatcher(None, answer, cluster_key).ratio() >= similarity_threshold:
                    answer_clusters[cluster_key] += 1  # Increment vote count
                    found_cluster = True
                    break
            # If no similar cluster found, create a new one
            if not found_cluster:
                answer_clusters[answer] = 1  # Initialize vote count

        if not answer_clusters:
            return pd.Series({
                "agg_answer": None,
                "answer_variants": 0
            })

        # Find the answer with the highest vote count
        best_answer = max(answer_clusters.keys(), key=answer_clusters.get)

        return pd.Series({
            "agg_answer": best_answer,
            "answer_variants": len(answer_clusters)
        })
    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)
                

'''
def aggregate_trivia(df: pd.DataFrame, group_col: str = "question_id", 
                               similarity_threshold: float = 0.8) -> pd.DataFrame:
    """
    Aggregate multi-sample TriviaQA results using vote count of answers and answer clustering.
    
    Args:
        df: DataFrame with parsed_answer column
        group_col: Column to group by (default: question_id)
        similarity_threshold: Threshold for considering answers similar (0.8 default)
    
    Returns:
        DataFrame with aggregated results containing:
        - agg_answer: The answer with highest total confidence
        - answer_variants: Number of unique answer variants
    """
    def agg_group(group):
        # Only consider rows with valid answers
        group_valid = group.dropna(subset=["parsed_answer"]).copy()
        group_valid["norm_answer"] = group_valid["parsed_answer"].apply(normalize_trivia_multi)
        group_valid = group_valid.dropna(subset=["norm_answer"])

        if group_valid.empty:
            return pd.Series({
                "agg_answer": None,
                "answer_variants": 0
            })

        # Cluster similar answers together
        answer_clusters = {}
        for _, row in group_valid.iterrows():
            answer = row["norm_answer"]
            
            # Find if this answer is similar to any existing cluster
            found_cluster = False
            for cluster_key in answer_clusters.keys():
                if SequenceMatcher(None, answer, cluster_key).ratio() >= similarity_threshold:
                    answer_clusters[cluster_key] += 1  # Increment vote count
                    found_cluster = True
                    break

            # If no similar cluster found, create a new one
            if not found_cluster:
                answer_clusters[answer] = 1  # Initialize vote count

        if not answer_clusters:
            return pd.Series({
                "agg_answer": None,
                "answer_variants": 0
            })

        # Find the answer with the highest vote count
        best_answer = max(answer_clusters.keys(), key=answer_clusters.get)

        return pd.Series({
            "agg_answer": best_answer,
            "answer_variants": len(answer_clusters)
        })

    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)


def evaluate_trivia_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, 
                              group_col: str = "question_id") -> tuple:
    """
    Evaluate aggregated TriviaQA results against ground truth.
    
    Args:
        df_agg: DataFrame with aggregated results (from aggregate_confidence_trivia)
        df_full: Original DataFrame with ground truth answers
        group_col: Column to group by (default: question_id)
    
    Returns:
        Tuple of (evaluation_df, accuracy_score)
    """
    # Get ground truth answers (one per group)
    gold = df_full.drop_duplicates(group_col)[[group_col, "answers"]].copy()
    
    def safe_parse_answers(val):
        """Converts stringified lists or values into real lists."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                return [val]
        return [val] if val else []
    
    gold["answer_list"] = gold["answers"].apply(safe_parse_answers)

    # Merge with aggregated results
    merged = df_agg.merge(gold, on=group_col, how="left")
    
    # Evaluate each aggregated answer against the gold answers
    def eval_row(row):
        pred = row["agg_answer"]
        golds = row["answer_list"]
        return evaluate_trivia_answer(pred, golds)
    
    merged["is_correct"] = merged.apply(eval_row, axis=1)
    accuracy = merged["is_correct"].mean()
    
    return merged, accuracy

def parse_aggregate_evaluate_trivia_multi(df: pd.DataFrame, group_col: str = "question_id",
                                        similarity_threshold: float = 0.8) -> tuple:
    """
    Complete pipeline for multi-sample TriviaQA: parse, aggregate, and evaluate.
    
    Args:
        df: DataFrame with model_output, answers, and grouping column
        group_col: Column to group by (default: question_id)
        similarity_threshold: Threshold for clustering similar answers
    
    Returns:
        Tuple of (parsed_df, aggregated_df, evaluation_df, accuracy_score)
    """
    # Step 1: Parse individual outputs
    df_parsed = parse_and_evaluate_trivia_multi(df, group_col)
    
    # Step 2: Aggregate by confidence with answer clustering
    df_agg = aggregate_trivia(df_parsed, group_col, similarity_threshold)
    
    # Step 3: Evaluate aggregated results
    eval_df, accuracy = evaluate_trivia_aggregated(df_agg, df_parsed, group_col)
    
    return df_parsed, df_agg, eval_df, accuracy
'''
    
# =====================
# MULTI-SAMPLE SQUAD PARSING & AGGREGATION
# =====================
def normalize_squad_multi(ans):
    """
    Normalize SQuAD answers for multi-sample aggregation.
    Returns normalized lowercase string for consistency.
    """
    if pd.isna(ans):
        return None
    ans = str(ans).strip().lower()
    ans = re.sub(r'[.,!?;:\"\'\(\)\[\]]', '', ans)
    ans = re.sub(r'\s+', ' ', ans).strip()
    return ans if ans else None

def parse_and_evaluate_squad_multi(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """
    Parse SQuAD outputs for multi-sample data and add parsed columns.
    Expects columns: model_output, answers, and a grouping column (default: question_id)
    Adds: parsed_answer
    """
    # Parse only if not already present
    if "parsed_answer" not in df.columns or "parsed_confidence" not in df.columns:
        parsed = df["model_output"].apply(parse_squad_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    
    return df

'''
def aggregate_confidence_squad(df: pd.DataFrame, group_col: str = "question_id", 
                              similarity_threshold: float = 0.8) -> pd.DataFrame:
    """
    Aggregate multi-sample SQuAD results using confidence-weighted voting with answer clustering.
    
    Args:
        df: DataFrame with parsed_answer and parsed_confidence columns
        group_col: Column to group by (default: question_id)
        similarity_threshold: Threshold for considering answers similar (0.8 default)
    
    Returns:
        DataFrame with aggregated results containing:
        - agg_answer: The answer with highest total confidence
        - agg_confidence: Relative confidence (0-100) of the chosen answer
        - answer_variants: Number of unique answer variants
    """
    def agg_group(group):
        # Only consider rows with both answer and confidence
        group_valid = group.dropna(subset=["parsed_answer", "parsed_confidence"]).copy()
        group_valid["norm_answer"] = group_valid["parsed_answer"].apply(normalize_squad_multi)
        group_valid = group_valid.dropna(subset=["norm_answer", "parsed_confidence"])

        if group_valid.empty:
            return pd.Series({
                "agg_answer": None, 
                "agg_confidence": 0,
                "answer_variants": 0
            })

        # Cluster similar answers together
        answer_clusters = {}
        for _, row in group_valid.iterrows():
            answer = row["norm_answer"]
            confidence = row["parsed_confidence"]
            
            # Find if this answer is similar to any existing cluster
            found_cluster = False
            for cluster_key in answer_clusters.keys():
                if SequenceMatcher(None, answer, cluster_key).ratio() >= similarity_threshold:
                    answer_clusters[cluster_key] += confidence
                    found_cluster = True
                    break
            
            # If no similar cluster found, create a new one
            if not found_cluster:
                answer_clusters[answer] = confidence
        
        if not answer_clusters:
            return pd.Series({
                "agg_answer": None, 
                "agg_confidence": 0,
                "answer_variants": 0
            })
        
        # Find the answer with highest total confidence
        best_answer = max(answer_clusters.keys(), key=answer_clusters.get)
        total_conf = sum(answer_clusters.values())
        rel_conf = answer_clusters[best_answer] / total_conf if total_conf > 0 else 0
        
        return pd.Series({
            "agg_answer": best_answer, 
            "agg_confidence": rel_conf * 100,
            "answer_variants": len(answer_clusters)
        })

    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)

def evaluate_squad_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, 
                             group_col: str = "question_id") -> tuple:
    """
    Evaluate aggregated SQuAD results against ground truth.
    
    Args:
        df_agg: DataFrame with aggregated results (from aggregate_confidence_squad)
        df_full: Original DataFrame with ground truth answers
        group_col: Column to group by (default: question_id)
    
    Returns:
        Tuple of (evaluation_df, accuracy_score)
    """
    # Get ground truth answers (one per group)
    gold = df_full.drop_duplicates(group_col)[[group_col, "answers"]].copy()
    
    def safe_parse_answers(val):
        """Converts stringified lists or values into real lists."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                return [val]
        return [val] if val else []
    
    gold["answer_list"] = gold["answers"].apply(safe_parse_answers)

    # Merge with aggregated results
    merged = df_agg.merge(gold, on=group_col, how="left")
    
    # Evaluate each aggregated answer against the gold answers
    def eval_row(row):
        pred = row["agg_answer"]
        golds = row["answer_list"]
        return evaluate_squad_answer(pred, golds)
    
    merged["is_correct"] = merged.apply(eval_row, axis=1)
    accuracy = merged["is_correct"].mean()
    
    return merged, accuracy

def parse_aggregate_evaluate_squad_multi(df: pd.DataFrame, group_col: str = "question_id",
                                        similarity_threshold: float = 0.8, 
                                        remove_unanswerable: bool = False) -> tuple:
    """
    Complete pipeline for multi-sample SQuAD: parse, aggregate, and evaluate.
    
    Args:
        df: DataFrame with model_output, answers, and grouping column
        group_col: Column to group by (default: question_id)
        similarity_threshold: Threshold for clustering similar answers
        remove_unanswerable: If True, removes unanswerable questions from the dataset
    
    Returns:
        Tuple of (parsed_df, aggregated_df, evaluation_df, accuracy_score)
    """
    # Step 1: Parse individual outputs
    df_parsed = parse_and_evaluate_squad_multi(df, group_col)
    
    # Remove unanswerable questions if requested
    if remove_unanswerable and "is_impossible" in df_parsed.columns:
        df_parsed = df_parsed[~df_parsed["is_impossible"]].copy()
        print(f"Removed {(df['is_impossible'] == True).sum()} unanswerable questions. {len(df_parsed)} questions remaining.")
    
    # Step 2: Aggregate by confidence with answer clustering
    df_agg = aggregate_confidence_squad(df_parsed, group_col, similarity_threshold)
    
    # Step 3: Evaluate aggregated results
    eval_df, accuracy = evaluate_squad_aggregated(df_agg, df_parsed, group_col)
    
    return df_parsed, df_agg, eval_df, accuracy
'''