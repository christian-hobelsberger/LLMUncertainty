# Import packages
import re
import json
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# =====================
# Normalization helpers
# =====================

def _strip_punct_spaces(s: str) -> str:
    """Lowercase, strip punctuation and extra spaces."""
    s = s.strip().lower()
    # Keep letters/numbers and spaces
    s = re.sub(r'[^0-9a-zäöüß ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_text_answer(s: Optional[str]) -> Optional[str]:
    """Lowercase, strip punctuation and extra spaces."""
    if s is None:
        return None
    return _strip_punct_spaces(str(s))

def normalize_bool_answer(s: Any) -> Optional[str]:
    """Normalize to "True", "False", or None."""
    if isinstance(s, bool):
        return "True" if s else "False"
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"true", "yes", "1"}:
        return "True"
    if t in {"false", "no", "0"}:
        return "False"
    return None

def normalize_gsm8k_answer(s: Any) -> Optional[str]:
    """Extract a plausible numeric answer (as string) from free text."""
    if s is None:
        return None
    text = str(s)
    # Look for an integer or decimal; prefer the last number in the string
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if not nums:
        return None
    return nums[-1]

def safe_parse_answers(x: Any) -> List[str]:
    """Return a list of gold answers from various formats (str, list, json string)."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, tuple):
        return [str(i) for i in list(x)]
    # Try JSON
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(i) for i in parsed]
                # some datasets store {"text": "..."} etc.
                if isinstance(parsed, dict):
                    vals = []
                    for v in parsed.values():
                        if isinstance(v, (list, tuple)):
                            vals.extend([str(i) for i in v])
                        else:
                            vals.append(str(v))
                    return vals
            except Exception:
                pass
    # Fallback: single string
    return [str(x)]

# =====================
# Universal answer parser (answers only)
# =====================

def universal_answer_parser(output: Any) -> Dict[str, Any]:
    """
    Robustly extract an 'answer' string from messy model outputs.
    Handles JSON blobs, "Answer: ...", "Final answer:", German variants ("Antwort:", "Lösung:"), and free text.
    Returns: {'answer': str | None}
    """
    if output is None:
        return {"answer": None}
    s = str(output).strip()
    if not s:
        return {"answer": None}

    # 0) Strip code fences for easier parsing
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`").strip()

    # 1) Try JSON objects anywhere in the string (including multiple glued objects)
    # Prefer the LAST object that contains an 'answer' key
    json_objs = re.findall(r'\{[^{}]*\}', s, flags=re.DOTALL)
    for jm in reversed(json_objs):
        try:
            obj = json.loads(jm)
            if isinstance(obj, dict):
                # Common answer keys
                for key in ["answer", "final_answer", "prediction", "predicted_answer", "antwort", "loesung", "lösung", "ergebnis"]:
                    if key in obj and isinstance(obj[key], (str, int, float, bool)):
                        val = obj[key]
                        if isinstance(val, bool):
                            val = "True" if val else "False"
                        else:
                            val = str(val)
                        if val.strip():
                            return {"answer": val.strip()}
        except Exception:
            continue

    # 2) Direct key-value lines like "Answer: ...", "Antwort: ...", "Final answer: ..."
    kv_patterns = [
        r'(?i)\bfinal\s*answer\s*[:=]\s*(.+)',
        r'(?i)\banswer\s*[:=]\s*(.+)',
        r'(?i)\bantwort\s*[:=]\s*(.+)',
        r'(?i)\blösung\s*[:=]\s*(.+)',
        r'(?i)\bloesung\s*[:=]\s*(.+)',
        r'(?i)\bergebnis\s*[:=]\s*(.+)',
        r'(?i)\bprediction\s*[:=]\s*(.+)',
        r'(?i)\bpredicted_answer\s*[:=]\s*(.+)',
        r'(?i)\bfinal\s*[:=]\s*(.+)',
        r'(?i)\bA\s*[:=]\s*(.+)',
        r'(?i)\bselected\s*option\s*[:=]\s*(.+)',
        r'(?i)\bthe\s+answer\s+is\s*(.+)',
    ]
    def _cleanup(val: str) -> str:
        # Cut at common trailing markers
        val = re.split(r'(?i)\b(confidence|score|prob(ability)?|explanation|reason|justification|rationale|analysis)\b', val)[0]
        val = re.split(r'```|###', val)[0]
        # Trim quotes and markdown bullets
        val = val.strip().strip('"').strip("'").strip("•-").strip()
        # Drop trailing punctuation-only
        return val.strip()

    for pat in kv_patterns:
        m = re.search(pat, s)
        if m:
            cand = _cleanup(m.group(1))
            if cand:
                return {"answer": cand}

    # 3) Try to capture quoted value after the word answer (Answer "X")
    m = re.search(r'(?i)\banswer\b[^"\n\r]*"([^"]+)"', s)
    if m and m.group(1).strip():
        return {"answer": m.group(1).strip()}

    # 4) Fallbacks: take last non-empty, content-like line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for ln in reversed(lines):
        # skip boilerplate
        if re.match(r'(?i)^(the\s*)?(final\s*)?answer\s*[:=]?\s*$', ln):
            continue
        if len(ln) >= 1:
            return {"answer": ln.strip(' "\'-•*')}

    # 5) As a last resort: first wordy chunk
    m = re.search(r'\b\w+(?:\s+\w+)*\b', s)
    if m:
        return {"answer": m.group(0).strip()}
    return {"answer": None}

def debug_preview_parsing(df: pd.DataFrame, output_col: str = "model_output", n: int = 10) -> pd.DataFrame:
    """
    Return a small dataframe preview with original output and the parsed answer for quick debugging.
    """
    sample = df[[output_col]].head(n).copy()
    sample["parsed"] = sample[output_col].apply(lambda x: universal_answer_parser(x).get("answer"))
    return sample

# =====================
# Task-specific parsers (answers only)
# =====================

def parse_trivia_output(output: Any) -> Dict[str, Any]:
    parsed = universal_answer_parser(output)
    ans = parsed.get('answer')
    return {"answer": normalize_text_answer(ans) if ans else None}

def parse_boolq_output(output: Any) -> Dict[str, Any]:
    parsed = universal_answer_parser(output)
    ans = parsed.get('answer')
    return {"answer": normalize_bool_answer(ans) if ans is not None else None}

def parse_gsm8k_output(output: Any) -> Dict[str, Any]:
    parsed = universal_answer_parser(output)
    ans = parsed.get('answer')
    return {"answer": normalize_gsm8k_answer(ans) if ans else None}

def parse_squad_output(output: Any) -> Dict[str, Any]:
    parsed = universal_answer_parser(output)
    ans = parsed.get('answer')
    return {"answer": normalize_text_answer(ans) if ans else None}

# =====================
# Row-wise evaluation helpers
# =====================

def evaluate_trivia_answer(predicted: Optional[str], gold_answers: List[str], threshold: float = 0.8) -> bool:
    """Exact match; if not, use fuzzy/substring heuristics."""
    if not predicted or not gold_answers:
        return False
    p = normalize_text_answer(predicted)
    for g in gold_answers:
        g_norm = normalize_text_answer(g)
        if p == g_norm:
            return True
        if len(p) > 2 and len(g_norm) > 2:
            if p in g_norm or g_norm in p:
                return True
            if SequenceMatcher(None, p, g_norm).ratio() >= threshold:
                return True
    return False

# =====================
# Parse + evaluate (row-wise)
# =====================

def parse_and_evaluate_boolq_multi(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    df = df.copy()
    if "parsed_answer" not in df.columns:
        parsed = df["model_output"].apply(parse_boolq_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    # Evaluate if gold present
    if "answer" in df.columns or "label" in df.columns:
        def _eval(row):
            pred = row.get("parsed_answer")
            gold = row.get("answer") if "answer" in row else row.get("label")
            gold_norm = normalize_bool_answer(gold)
            pred_norm = normalize_bool_answer(pred)
            return bool(pred_norm and gold_norm and pred_norm == gold_norm)
        df["is_correct"] = df.apply(_eval, axis=1)
    return df

def parse_and_evaluate_trivia(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    df = df.copy()
    if "parsed_answer" not in df.columns:
        parsed = df["model_output"].apply(parse_trivia_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    if "answers" in df.columns:
        df["is_correct"] = [
            evaluate_trivia_answer(p, safe_parse_answers(g))
            for p, g in zip(df["parsed_answer"], df["answers"])
        ]
    return df

def parse_and_evaluate_gsm8k_multi(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    df = df.copy()
    if "parsed_answer" not in df.columns:
        parsed = df["model_output"].apply(parse_gsm8k_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    if "answer" in df.columns:
        df["is_correct"] = [
            (normalize_gsm8k_answer(p) == normalize_gsm8k_answer(g)) if (p is not None and g is not None) else False
            for p, g in zip(df["parsed_answer"], df["answer"])
        ]
    return df

def parse_and_evaluate_squad_multi(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    df = df.copy()
    if "parsed_answer" not in df.columns:
        parsed = df["model_output"].apply(parse_squad_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    if "answers" in df.columns:
        df["is_correct"] = [
            evaluate_trivia_answer(p, safe_parse_answers(g))
            for p, g in zip(df["parsed_answer"], df["answers"])
        ]
    return df

# =====================
# Aggregation (answers only)
# =====================

def aggregate_answers_boolq(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """Unweighted majority vote over parsed answers. Returns [group_col, 'agg_answer']."""
    def agg_group(group):
        answers = group['parsed_answer'].dropna().astype(str)
        if len(answers) == 0:
            return pd.Series({'agg_answer': None})
        counts = answers.value_counts()
        return pd.Series({'agg_answer': counts.idxmax()})
    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)

def aggregate_answers_trivia(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """Normalized string majority vote. Returns [group_col, 'agg_answer', 'answer_variants']"""
    def norm(x):
        return normalize_text_answer(x) if pd.notna(x) else None
    def agg_group(group):
        norm_ans = group['parsed_answer'].dropna().map(norm).dropna()
        if len(norm_ans) == 0:
            return pd.Series({'agg_answer': None, 'answer_variants': 0})
        counts = norm_ans.value_counts()
        top_norm = counts.idxmax()
        rep = group.loc[group['parsed_answer'].map(norm) == top_norm, 'parsed_answer']
        rep_answer = rep.iloc[0] if len(rep) else top_norm
        return pd.Series({'agg_answer': rep_answer, 'answer_variants': counts.size})
    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)

def aggregate_answers_gsm8k(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """Normalized numeric majority vote. Returns [group_col, 'agg_answer', 'answer_variants']"""
    def norm(x):
        return normalize_gsm8k_answer(x) if pd.notna(x) else None
    def agg_group(group):
        norm_ans = group['parsed_answer'].dropna().map(norm).dropna()
        if len(norm_ans) == 0:
            return pd.Series({'agg_answer': None, 'answer_variants': 0})
        counts = norm_ans.value_counts()
        top_norm = counts.idxmax()
        rep = group.loc[group['parsed_answer'].map(norm) == top_norm, 'parsed_answer']
        rep_answer = rep.iloc[0] if len(rep) else top_norm
        return pd.Series({'agg_answer': rep_answer, 'answer_variants': counts.size})
    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)

def aggregate_answers_squad(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    """Normalized string majority vote. Returns [group_col, 'agg_answer', 'answer_variants']"""
    def norm(x):
        return normalize_text_answer(x) if pd.notna(x) else None
    def agg_group(group):
        norm_ans = group['parsed_answer'].dropna().map(norm).dropna()
        if len(norm_ans) == 0:
            return pd.Series({'agg_answer': None, 'answer_variants': 0})
        counts = norm_ans.value_counts()
        top_norm = counts.idxmax()
        rep = group.loc[group['parsed_answer'].map(norm) == top_norm, 'parsed_answer']
        rep_answer = rep.iloc[0] if len(rep) else top_norm
        return pd.Series({'agg_answer': rep_answer, 'answer_variants': counts.size})
    return df.groupby(group_col, as_index=False).apply(agg_group).reset_index(drop=True)

# =====================
# Evaluation of aggregated predictions
# =====================

def evaluate_boolq_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, group_col: str = "question_id"):
    """
    Evaluate aggregated BoolQ predictions by comparing 'agg_answer' to normalized gold. 
    Returns (evaluation_df, accuracy).
    """
    gold = df_full.drop_duplicates(group_col)[[group_col, "answer"]].copy()
    gold["answer_norm"] = gold["answer"].apply(normalize_bool_answer)
    merged = df_agg.merge(gold, on=group_col, how="left")
    merged["is_correct"] = merged["agg_answer"] == merged["answer_norm"]
    acc = float(merged["is_correct"].mean()) if len(merged) else 0.0
    return merged, acc

def evaluate_trivia_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, group_col: str = "question_id"):
    """
    Evaluate aggregated Trivia predictions using the same heuristics as per-row evaluation. 
    Returns (evaluation_df, accuracy).
    """
    gold = df_full.drop_duplicates(group_col)[[group_col, "answers"]].copy()
    merged = df_agg.merge(gold, on=group_col, how="left")
    merged["is_correct"] = [
        evaluate_trivia_answer(p, safe_parse_answers(g))
        for p, g in zip(merged["agg_answer"], merged["answers"])
    ]
    acc = float(merged["is_correct"].mean()) if len(merged) else 0.0
    return merged, acc

def evaluate_gsm8k_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, group_col: str = "question_id"):
    """
    Evaluate aggregated GSM8K predictions by numeric-string equality after normalization. 
    Returns (evaluation_df, accuracy).
    """
    gold = df_full.drop_duplicates(group_col)[[group_col, "answer"]].copy()
    gold["answer_norm"] = gold["answer"].apply(normalize_gsm8k_answer)
    merged = df_agg.merge(gold, on=group_col, how="left")
    merged["is_correct"] = merged["agg_answer"].map(normalize_gsm8k_answer) == merged["answer_norm"]
    acc = float(merged["is_correct"].mean()) if len(merged) else 0.0
    return merged, acc

def evaluate_squad_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, group_col: str = "question_id"):
    """
    Evaluate aggregated SQuAD-style predictions and keep 'is_impossible' if present.

    Args:
        df_agg: DataFrame with aggregated results (columns include [group_col, 'agg_answer', ...])
        df_full: Original DataFrame (must contain [group_col, 'answers']; may contain 'is_impossible')
        group_col: Question id column (default: 'question_id')

    Returns:
        (evaluation_df, accuracy)
        evaluation_df columns: [group_col, 'agg_answer', 'answers', 'is_impossible?' (optional), 'is_correct']
    """
    # One gold record per question
    gold = df_full.drop_duplicates(group_col)[[group_col, "answers"]].copy()
    # Merge predictions with gold
    merged = df_agg.merge(gold, on=group_col, how="left")
    # Keep is_impossible if present
    if "is_impossible" in df_full.columns:
        flags = df_full[[group_col, "is_impossible"]].drop_duplicates(group_col)
        eval_df = eval_df.merge(flags, on=group_col, how="left")
    # Evaluate correctness
    merged["is_correct"] = [
        evaluate_trivia_answer(p, safe_parse_answers(g))
        for p, g in zip(merged["agg_answer"], merged["answers"])
    ]
    
    acc = float(merged["is_correct"].mean()) if len(merged) else 0.0
    return merged, acc

# =====================
# Pipelines
# =====================

def parse_aggregate_evaluate_boolq_multi(df: pd.DataFrame, group_col: str = "question_id"):
    """
    One-call pipeline for BoolQ: parse rows → aggregate by majority vote -> evaluate aggregates. 
    Returns (df_parsed, df_agg, eval_df, accuracy).
    """
    df_parsed = parse_and_evaluate_boolq_multi(df, group_col)
    df_agg = aggregate_answers_boolq(df_parsed, group_col=group_col)
    eval_df, accuracy = evaluate_boolq_aggregated(df_agg, df_parsed, group_col=group_col)
    return df_parsed, df_agg, eval_df, accuracy

def parse_aggregate_evaluate_trivia(df: pd.DataFrame, group_col: str = "question_id"):
    """
    One-call pipeline for TriviaQA: parse rows → aggregate by majority vote -> evaluate aggregates. 
    Returns (df_parsed, df_agg, eval_df, accuracy).
    """
    df_parsed = parse_and_evaluate_trivia(df, group_col)
    df_agg = aggregate_answers_trivia(df_parsed, group_col=group_col)
    eval_df, accuracy = evaluate_trivia_aggregated(df_agg, df_parsed, group_col=group_col)
    return df_parsed, df_agg, eval_df, accuracy

def parse_aggregate_evaluate_gsm8k_multi(df: pd.DataFrame, group_col: str = "question_id"):
    """
    One-call pipeline for GSM8K: parse rows -> aggregate by majority vote → evaluate aggregates. 
    Returns (df_parsed, df_agg, accuracy).
    """
    df_parsed = parse_and_evaluate_gsm8k_multi(df, group_col)
    df_agg = aggregate_answers_gsm8k(df_parsed, group_col=group_col)
    eval_df, accuracy = evaluate_gsm8k_aggregated(df_agg, df_parsed, group_col=group_col)
    return df_parsed, df_agg, eval_df, accuracy

def parse_aggregate_evaluate_squad_multi(df: pd.DataFrame, group_col: str = "question_id", remove_unanswerable: bool = True):
    """
    One-call pipeline for SQuAD: parse rows -> (optional) filter unanswerable -> aggregate by majority vote -> evaluate. 
    Returns (df_parsed, df_agg, eval_df, accuracy).
    """
    df_parsed = parse_and_evaluate_squad_multi(df, group_col)
    if remove_unanswerable and "is_impossible" in df_parsed.columns:
        df_parsed = df_parsed[~df_parsed["is_impossible"]].copy()
    df_agg = aggregate_answers_squad(df_parsed, group_col=group_col)
    eval_df, accuracy = evaluate_squad_aggregated(df_agg, df_parsed, group_col=group_col)
    return df_parsed, df_agg, eval_df, accuracy


# =====================
# Flexible variants (explicitly choose the model output column)
# =====================

def parse_and_evaluate_squad_multi_flex(df: pd.DataFrame, output_col: str = "model_output", group_col: str = "question_id") -> pd.DataFrame:
    df = df.copy()
    if "parsed_answer" not in df.columns:
        parsed = df[output_col].apply(parse_squad_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    if "answers" in df.columns:
        df["is_correct"] = [
            evaluate_trivia_answer(p, safe_parse_answers(g))
            for p, g in zip(df["parsed_answer"], df["answers"])
        ]
    return df

def parse_aggregate_evaluate_squad_multi_flex(df: pd.DataFrame, output_col: str = "model_output", group_col: str = "question_id", remove_unanswerable: bool = True):
    df_parsed = parse_and_evaluate_squad_multi_flex(df, output_col=output_col, group_col=group_col)
    if remove_unanswerable and "is_impossible" in df_parsed.columns:
        df_parsed = df_parsed[~df_parsed["is_impossible"]].copy()
    df_agg = aggregate_answers_squad(df_parsed, group_col=group_col)
    eval_df, accuracy = evaluate_squad_aggregated(df_agg, df_parsed, group_col=group_col)
    return df_parsed, df_agg, eval_df, accuracy
