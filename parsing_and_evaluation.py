import re
import json
import ast
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# =====================
# BOOLQ PARSING & EVAL
# =====================
def normalize_bool_answer(answer) -> str:
    """
    Normalize various yes/no/true/false cases into 'True' or 'False'.
    Returns None if not recognized.
    """
    if answer is None:
        return None
    
    # Handle boolean inputs directly
    if isinstance(answer, bool):
        return "True" if answer else "False"
    
    # Handle string inputs
    if not isinstance(answer, str):
        answer = str(answer)
    
    answer = answer.strip().lower()
    if answer in {"true", "yes", "1"}:
        return "True"
    elif answer in {"false", "no", "0"}:
        return "False"
    return None

def parse_boolq_output(output: str) -> dict:
    """
    Parses BoolQ model output to extract a single answer and confidence score.
    Returns dict with keys 'answer' and 'confidence'.
    """
    try:
        output = str(output).strip()
        if not output:
            return {"answer": None, "confidence": None}
        # Try JSON
        json_match = re.search(r'\{.*?\}', output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and "answer" in parsed:
                    norm_answer = normalize_bool_answer(str(parsed["answer"]))
                    confidence = parsed.get("confidence", None)
                    return {"answer": norm_answer, "confidence": confidence}
            except json.JSONDecodeError:
                pass
        # Fallback regex
        answer_match = re.search(r'Answer\s*[:=]?\s*(True|False|Yes|No)', output, re.IGNORECASE)
        conf_match = re.search(r'Confidence\s*[:=]?\s*(\d+)', output, re.IGNORECASE)
        answer = normalize_bool_answer(answer_match.group(1)) if answer_match else None
        confidence = int(conf_match.group(1)) if conf_match else None
        return {"answer": answer, "confidence": confidence}
    except Exception:
        return {"answer": None, "confidence": None}

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
def normalize_trivia_answer(answer: str) -> str:
    """
    Normalize trivia answers: lowercase, remove punctuation, normalize whitespace.
    """
    if not answer:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r'[.,!?;:\"\'\(\)\[\]]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer

def parse_trivia_output(output: str) -> dict:
    """
    Parses model output for answer/confidence, using multiple strategies.
    Returns dict with keys 'answer' and 'confidence'.
    """
    try:
        output = str(output).strip()
        if not output:
            return {"answer": None, "confidence": None}
        # Try JSON
        json_match = re.search(r'\{.*?\}', output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = normalize_trivia_answer(parsed["answer"])
                    confidence = parsed.get("confidence", None)
                    return {"answer": answer if answer else None, "confidence": confidence}
            except json.JSONDecodeError:
                pass
        # Fallback regex
        answer_match = re.search(r'Answer\s*[:=]?\s*(.+?)(?:\n|Confidence|$)', output, re.IGNORECASE | re.DOTALL)
        conf_match = re.search(r'Confidence\s*[:=]?\s*(\d+)', output, re.IGNORECASE)
        if answer_match:
            answer = normalize_trivia_answer(answer_match.group(1))
        else:
            first_sentence = re.split(r'[.!?\n]', output)[0]
            answer = normalize_trivia_answer(first_sentence[:50])
        confidence = int(conf_match.group(1)) if conf_match else None
        return {"answer": answer if answer else None, "confidence": confidence}
    except Exception:
        return {"answer": None, "confidence": None}

def evaluate_trivia_answer(predicted: str, gold_answers: list, threshold: float = 0.8) -> bool:
    """
    Compares a predicted answer to a list of gold answers with fuzzy and substring matching.
    """
    if not predicted or not gold_answers:
        return False
    predicted = normalize_trivia_answer(predicted)
    for gold in gold_answers:
        gold_norm = normalize_trivia_answer(str(gold))
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
def normalize_squad_answer(answer: str) -> str:
    """
    Normalize SQuAD answers: lowercase, remove punctuation, normalize whitespace.
    Similar to TriviaQA normalization but handles SQuAD-specific cases.
    """
    if not answer:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r'[.,!?;:\"\'\(\)\[\]]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer

def parse_squad_output(output: str) -> dict:
    """
    Parses model output for SQuAD answer/confidence, using multiple strategies.
    Returns dict with keys 'answer' and 'confidence'.
    """
    try:
        output = str(output).strip()
        if not output:
            return {"answer": None, "confidence": None}
        
        # Try JSON
        json_match = re.search(r'\{.*?\}', output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = normalize_squad_answer(parsed["answer"])
                    confidence = parsed.get("confidence", None)
                    return {"answer": answer if answer else None, "confidence": confidence}
            except json.JSONDecodeError:
                pass
        
        # Fallback regex
        answer_match = re.search(r'Answer\s*[:=]?\s*(.+?)(?:\n|Confidence|$)', output, re.IGNORECASE | re.DOTALL)
        conf_match = re.search(r'Confidence\s*[:=]?\s*(\d+)', output, re.IGNORECASE)
        
        if answer_match:
            answer = normalize_squad_answer(answer_match.group(1))
        else:
            # Use first sentence chunk if all else fails
            first_sentence = re.split(r'[.!?\n]', output)[0]
            answer = normalize_squad_answer(first_sentence[:50])
        
        confidence = int(conf_match.group(1)) if conf_match else None
        return {"answer": answer if answer else None, "confidence": confidence}
    except Exception:
        return {"answer": None, "confidence": None}

def evaluate_squad_answer(predicted: str, gold_answers: list, threshold: float = 0.8) -> bool:
    """
    Compares a predicted answer to a list of gold answers with fuzzy and substring matching.
    Similar to TriviaQA evaluation but handles SQuAD-specific cases including unanswerable questions.
    """
    if not predicted or not gold_answers:
        return False
    
    predicted = normalize_squad_answer(predicted)
    
    # Handle unanswerable questions - check if predicted answer indicates "no answer"
    no_answer_indicators = {"", "no answer", "unanswerable", "cannot answer", "not answerable", "no", "none"}
    if predicted in no_answer_indicators:
        # Check if any gold answer is also empty/unanswerable
        return any(normalize_squad_answer(str(gold)) in no_answer_indicators for gold in gold_answers)
    
    for gold in gold_answers:
        gold_norm = normalize_squad_answer(str(gold))
        
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
