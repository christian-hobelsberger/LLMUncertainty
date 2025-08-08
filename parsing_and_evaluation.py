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
def universal_answer_confidence_parser(output: str) -> Dict[str, Any]:
    """
    Universal parser that handles various answer/confidence formats robustly.
    
    Handles patterns like:
    - {"answer": "text", "confidence": 90}
    - {"answer": "text", "confidence": "90%"}
    - Answer: text\nconfidence: 90
    - Answer: text.\nconfidence: 50}"}{"answer": "text", "confidence": 100}
    - Multiple JSON objects in sequence
    - Malformed JSON with missing brackets
    - Various confidence formats (with/without %, as string/int)
    
    Returns: {"answer": str|None, "confidence": int|None}
    """
    if not output or pd.isna(output):
        return {"answer": None, "confidence": None}
    
    output = str(output).strip()
    if not output:
        return {"answer": None, "confidence": None}
    
    # Strategy 1: Try to find valid JSON objects (including multiple)
    json_matches = re.findall(r'\{[^{}]*\}', output, re.DOTALL)
    for json_match in json_matches:
        try:
            parsed = json.loads(json_match)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = str(parsed.get("answer", "")).strip()
                confidence = _extract_confidence_value(parsed.get("confidence"))
                if answer:  # Only return if we have a non-empty answer
                    return {"answer": answer, "confidence": confidence}
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Strategy 2: Try to fix malformed JSON (missing brackets, trailing characters)
    # Look for patterns like: "answer": "text", "confidence": 90}
    json_pattern = r'"answer"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([^}]+)'
    json_match = re.search(json_pattern, output, re.IGNORECASE | re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()
        confidence = _extract_confidence_value(json_match.group(2))
        if answer:
            return {"answer": answer, "confidence": confidence}
    
    # Strategy 3: Look for Answer:/Confidence: patterns
    answer_patterns = [
        r'Answer\s*[:=]?\s*([^\n\r]+?)(?:\s*(?:confidence|Confidence)|\s*$)',
        r'answer\s*[:=]?\s*([^\n\r]+?)(?:\s*(?:confidence|Confidence)|\s*$)',
        r'(?:^|\n)\s*([^\n\r]+?)\s*(?:confidence|Confidence)',  # Answer before confidence
    ]
    
    confidence_patterns = [
        r'(?:confidence|Confidence)\s*[:=]?\s*(\d+\.?\d*)\s*%?',
        r'(?:confidence|Confidence)\s*[:=]?\s*"?(\d+\.?\d*)%?"?',
    ]
    
    answer = None
    confidence = None
    
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
    
    # Extract confidence
    for pattern in confidence_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            confidence = _extract_confidence_value(match.group(1))
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
    
    return {"answer": answer, "confidence": confidence}

def _extract_confidence_value(conf_str: Any) -> Optional[int]:
    """
    Extract numeric confidence value from various formats.
    Handles: 90, "90", "90%", 90%, 0.9 (converts to 90), etc.
    """
    if conf_str is None:
        return None
    
    if isinstance(conf_str, (int, float)):
        val = float(conf_str)
        return int(val) if val >= 1 else int(val * 100)  # Handle 0.9 -> 90
    
    if isinstance(conf_str, str):
        # Remove quotes, %, and whitespace
        cleaned = re.sub(r'["%\s}]', '', conf_str)
        if not cleaned:
            return None
        
        try:
            val = float(cleaned)
            return int(val) if val >= 1 else int(val * 100)  # Handle 0.9 -> 90
        except ValueError:
            # Try to extract first number from string
            numbers = re.findall(r'\d+\.?\d*', cleaned)
            if numbers:
                val = float(numbers[0])
                return int(val) if val >= 1 else int(val * 100)
    
    return None

# =====================
# DATASET-SPECIFIC NORMALIZERS
# =====================
def normalize_bool_answer(answer) -> str:
    """Normalize various yes/no/true/false cases into 'True' or 'False'."""
    if answer is None:
        return None
    
    if isinstance(answer, bool):
        return "True" if answer else "False"
    
    if not isinstance(answer, str):
        answer = str(answer)
    
    answer = answer.strip().lower()
    if answer in {"true", "yes", "1"}:
        return "True"
    elif answer in {"false", "no", "0"}:
        return "False"
    return None

def normalize_text_answer(answer: str) -> str:
    """Normalize text answers: lowercase, remove punctuation, normalize whitespace."""
    if not answer:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r'[.,!?;:\"\'\(\)\[\]]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer

# =====================
# DATASET-SPECIFIC PARSING FUNCTIONS
# =====================
def parse_boolq_output(output: str) -> dict:
    """
    Parses BoolQ model output using universal parser, then normalizes for boolean answers.
    Returns dict with keys 'answer' and 'confidence'.
    """
    parsed = universal_answer_confidence_parser(output)
    answer = normalize_bool_answer(parsed.get("answer")) if parsed.get("answer") else None
    return {"answer": answer, "confidence": parsed.get("confidence")}
    if answer in {"true", "yes", "1"}:
        return "True"
    elif answer in {"false", "no", "0"}:
        return "False"
    return None

def parse_boolq_output(output: str) -> dict:
    """
    Parses BoolQ model output using universal parser, then normalizes for boolean answers.
    Returns dict with keys 'answer' and 'confidence'.
    """
    parsed = universal_answer_confidence_parser(output)
    answer = normalize_bool_answer(parsed.get("answer")) if parsed.get("answer") else None
    return {"answer": answer, "confidence": parsed.get("confidence")}

def evaluate_boolq_row(row) -> bool:
    """
    Evaluates a single row for BoolQ, comparing parsed_answer to answer (ground truth).
    Returns True if normalized answers match, False if not, np.nan if either is missing.
    """
    pred = row.get("parsed_answer")
    gold = row.get("answer")
    
    # Handle both string and boolean inputs for gold standard
    if isinstance(gold, bool):
        norm_gold = "True" if gold else "False"
    else:
        norm_gold = normalize_bool_answer(gold)
    
    # Handle parsed answer (should be string from parsing)
    norm_pred = normalize_bool_answer(pred)
    
    if norm_pred is None or norm_gold is None:
        return np.nan
    return norm_pred == norm_gold

def parse_and_evaluate_boolq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses BoolQ outputs and evaluates correctness.
    Expects columns: model_output, answer
    Adds: parsed_answer, parsed_confidence, is_correct
    """
    parsed = df["model_output"].apply(parse_boolq_output)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
    df["is_correct"] = df.apply(evaluate_boolq_row, axis=1)
    return df

# =====================
# TRIVIAQA PARSING & EVAL
# =====================
# normalize_trivia_answer replaced by normalize_text_answer

def parse_trivia_output(output: str) -> dict:
    """
    Parses TriviaQA model output using universal parser, then normalizes for text answers.
    Returns dict with keys 'answer' and 'confidence'.
    """
    parsed = universal_answer_confidence_parser(output)
    answer = normalize_text_answer(parsed.get("answer")) if parsed.get("answer") else None
    return {"answer": answer, "confidence": parsed.get("confidence")}

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
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
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
# MULTI-SAMPLE BOOLQ PARSING & AGGREGATION
# =====================
def normalize_boolq_multi(ans):
    """
    Normalize BoolQ answers for multi-sample aggregation.
    Returns 'true' or 'false' in lowercase for consistency.
    """
    if pd.isna(ans):
        return None
    ans = str(ans).strip().lower()
    if ans in {"true", "yes"}:
        return "true"
    if ans in {"false", "no"}:
        return "false"
    return ans

def parse_and_evaluate_boolq_multi(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """
    Parse BoolQ outputs for multi-sample data and add parsed columns.
    Expects columns: model_output, answer, and a grouping column (default: question_id)
    Adds: parsed_answer, parsed_confidence
    """
    # Parse only if not already present
    if "parsed_answer" not in df.columns or "parsed_confidence" not in df.columns:
        parsed = df["model_output"].apply(parse_boolq_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
        df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
    else:
        # Ensure parsed_confidence is numeric if it already exists
        df["parsed_confidence"] = pd.to_numeric(df["parsed_confidence"], errors="coerce")
    
    return df

def aggregate_confidence_boolq(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """
    Aggregate multi-sample BoolQ results using confidence-weighted voting.
    
    Args:
        df: DataFrame with parsed_answer and parsed_confidence columns
        group_col: Column to group by (default: question_id)
    
    Returns:
        DataFrame with aggregated results containing:
        - agg_answer: The answer with highest total confidence
        - agg_confidence: Relative confidence (0-100) of the chosen answer
    """
    def agg_group(group):
        # Only consider rows with both answer and confidence
        group_valid = group.dropna(subset=["parsed_answer", "parsed_confidence"]).copy()
        group_valid["norm_answer"] = group_valid["parsed_answer"].apply(normalize_boolq_multi)
        group_valid = group_valid.dropna(subset=["norm_answer", "parsed_confidence"])

        if group_valid.empty:
            return pd.Series({"agg_answer": None, "agg_confidence": 0})

        # Sum confidence scores for each answer
        conf_sum = group_valid.groupby("norm_answer")["parsed_confidence"].sum()
        best_ans = conf_sum.idxmax()
        total_conf = group_valid["parsed_confidence"].sum()
        rel_conf = conf_sum[best_ans] / total_conf if total_conf > 0 else 0
        
        return pd.Series({
            "agg_answer": best_ans, 
            "agg_confidence": rel_conf * 100
        })

    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)

def evaluate_boolq_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, group_col: str = "question_id") -> tuple:
    """
    Evaluate aggregated BoolQ results against ground truth.
    
    Args:
        df_agg: DataFrame with aggregated results (from aggregate_confidence_boolq)
        df_full: Original DataFrame with ground truth answers
        group_col: Column to group by (default: question_id)
    
    Returns:
        Tuple of (evaluation_df, accuracy_score)
    """
    # Get ground truth answers (one per group)
    gold = df_full.drop_duplicates(group_col)[[group_col, "answer"]].copy()
    gold["answer_norm"] = gold["answer"].apply(normalize_boolq_multi)

    # Merge with aggregated results
    merged = df_agg.merge(gold, on=group_col, how="left")
    merged["is_correct"] = merged["agg_answer"] == merged["answer_norm"]
    accuracy = merged["is_correct"].mean()
    
    return merged, accuracy

def parse_aggregate_evaluate_boolq_multi(df: pd.DataFrame, group_col: str = "question_id") -> tuple:
    """
    Complete pipeline for multi-sample BoolQ: parse, aggregate, and evaluate.
    
    Args:
        df: DataFrame with model_output, answer, and grouping column
        group_col: Column to group by (default: question_id)
    
    Returns:
        Tuple of (parsed_df, aggregated_df, evaluation_df, accuracy_score)
    """
    # Step 1: Parse individual outputs
    df_parsed = parse_and_evaluate_boolq_multi(df, group_col)
    
    # Step 2: Aggregate by confidence
    df_agg = aggregate_confidence_boolq(df_parsed, group_col)
    
    # Step 3: Evaluate aggregated results
    eval_df, accuracy = evaluate_boolq_aggregated(df_agg, df_parsed, group_col)
    
    return df_parsed, df_agg, eval_df, accuracy

# =====================
# SQUAD PARSING & EVAL
# =====================
# normalize_squad_answer replaced by normalize_text_answer

def parse_squad_output(output: str) -> dict:
    """
    Parses SQuAD model output using universal parser, then normalizes for text answers.
    Returns dict with keys 'answer' and 'confidence'.
    """
    # Force everything into a string
    if not isinstance(output, str):
        output = str(output)

    parsed = universal_answer_confidence_parser(output)
    answer = normalize_text_answer(parsed.get("answer")) if parsed.get("answer") else None
    return {"answer": answer, "confidence": parsed.get("confidence")}

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
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
    
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
    Adds: parsed_answer, parsed_confidence
    """
    # Parse only if not already present
    if "parsed_answer" not in df.columns or "parsed_confidence" not in df.columns:
        parsed = df["model_output"].apply(parse_trivia_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
        df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
    else:
        # Ensure parsed_confidence is numeric if it already exists
        df["parsed_confidence"] = pd.to_numeric(df["parsed_confidence"], errors="coerce")
    
    return df

def aggregate_confidence_trivia(df: pd.DataFrame, group_col: str = "question_id", 
                               similarity_threshold: float = 0.8) -> pd.DataFrame:
    """
    Aggregate multi-sample TriviaQA results using confidence-weighted voting with answer clustering.
    
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
        group_valid["norm_answer"] = group_valid["parsed_answer"].apply(normalize_trivia_multi)
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
    df_agg = aggregate_confidence_trivia(df_parsed, group_col, similarity_threshold)
    
    # Step 3: Evaluate aggregated results
    eval_df, accuracy = evaluate_trivia_aggregated(df_agg, df_parsed, group_col)
    
    return df_parsed, df_agg, eval_df, accuracy

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
    Adds: parsed_answer, parsed_confidence
    """
    # Parse only if not already present
    if "parsed_answer" not in df.columns or "parsed_confidence" not in df.columns:
        parsed = df["model_output"].apply(parse_squad_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
        df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
    else:
        # Ensure parsed_confidence is numeric if it already exists
        df["parsed_confidence"] = pd.to_numeric(df["parsed_confidence"], errors="coerce")
    
    return df

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
