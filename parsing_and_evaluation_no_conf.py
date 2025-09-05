# parsing_and_evaluation_no_conf2.py – confidence-free parsing + correctness (exact + fuzzy)
# ---------------------------------------------------------------------------------
# Robust parsing (inkl. mehrzeilige Answer-Blöcke), Gold-Parsing (JSON + Python-Literal),
# Exact+Fuzzy Matching, optionale Question-Aware-Extraktion und Majority Vote.
# ---------------------------------------------------------------------------------

from __future__ import annotations

import re
import json
import ast
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# =====================
# Normalization helpers
# =====================

def _strip_punct_spaces(s: str) -> str:
    """Lowercase, strip punctuation (keep letters/numbers/space), collapse whitespace.
    Uses stdlib ``re`` (no Unicode ``\p{..}``).
    """
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text_answer(s: Any) -> Optional[str]:
    if s is None:
        return None
    s_norm = _strip_punct_spaces(str(s))
    return s_norm if s_norm else None


def normalize_bool_answer(s: Any) -> Optional[str]:
    if isinstance(s, bool):
        return "True" if s else "False"
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"true", "yes", "y", "1"}:
        return "True"
    if t in {"false", "no", "n", "0"}:
        return "False"
    return None


# =====================
# Gold answers parser (robust to JSON *and* Python-list strings)
# =====================

def safe_parse_answers(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, tuple):
        return [str(i) for i in x]
    if isinstance(x, dict):
        vals: List[str] = []
        for v in x.values():
            if isinstance(v, (list, tuple)):
                vals.extend([str(i) for i in v])
            else:
                vals.append(str(v))
        return vals
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            parsed = None
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = None
            if isinstance(parsed, list):
                return [str(i) for i in parsed]
            if isinstance(parsed, dict):
                vals: List[str] = []
                for v in parsed.values():
                    if isinstance(v, (list, tuple)):
                        vals.extend([str(i) for i in v])
                    else:
                        vals.append(str(v))
                return vals
        if "|" in s:
            return [i.strip() for i in s.split("|") if i.strip()]
        if ";" in s:
            return [i.strip() for i in s.split(";") if i.strip()]
        return [s]
    return [str(x)]


# =====================
# Universal answer parser (answers only)
# =====================

PLACEHOLDERS = {"<string>", "string", "<number>", "number", "<boolean>", "boolean"}
INVALID_ANSWER_PHRASES = {"see explanation", "cannot determine", "unknown", "n/a", "not sure", "i don't know", "none"}


def _is_placeholder(val: str) -> bool:
    v = str(val).strip().lower()
    return (v in PLACEHOLDERS) or (v in INVALID_ANSWER_PHRASES)


def universal_answer_parser(output: Any) -> Dict[str, Any]:
    """Mehrzeiliges Answer-Harvesting + JSON/Key-Value-Fallbacks."""
    if output is None or (isinstance(output, float) and pd.isna(output)):
        return {"answer": None}
    s = str(output).strip()
    if not s:
        return {"answer": None}

    def _cleanup(val: str) -> str:
        val = re.split(r"(?i)\b(confidence|score|prob(ability)?|explanation|reason|justification|rationale|analysis)\b", val)[0]
        val = re.split(r"```|###", val)[0]
        val = re.sub(r"\s*\n\s*", " ", val)
        val = val.strip().strip('"').strip("'").strip("•-").strip()
        return val

    def _is_stop_marker(line: str) -> bool:
        return bool(re.match(r'(?is)^\s*(?:question\s*:|(?:final\s*)?answer\s*:|```|###)\b', line))

    def _is_noise_line(line: str) -> bool:
        t = line.strip()
        if not t:
            return True
        if t.startswith("{") or t.startswith("["):
            return True
        if _is_placeholder(t):
            return True
        return False

    lines = s.splitlines()

    # 0) letzte "Answer:"-Zeile finden und Folgezeilen ernten (Noise überspringen)
    last_idx = -1
    last_first = ""
    for i, line in enumerate(lines):
        m = re.match(r'(?is)^\s*(?:final\s*)?answer\s*[:=]\s*(.*)\s*$', line)
        if m:
            last_idx = i
            last_first = m.group(1).strip()

    # Helper: Stop-/Noise-Logik
    def _is_stop_marker(line: str) -> bool:
        # Nur echte Blockwechsel stoppen: neue Question:, Codefence.  <<< WICHTIG: 'answer:' NICHT mehr als Stop!
        return bool(re.match(r'(?is)^\s*(?:question\s*:|```|###)\b', line))

    def _is_noise_line(line: str) -> bool:
        t = line.strip()
        if not t:
            return True
        # JSON-/Dict-/Array-Fetzen: ignorieren, aber NICHT stoppen
        if t.startswith("{") or t.startswith("[") or t.startswith("}"):
            return True
        # Inline-JSON "answer": "<string>" -> ignorieren
        if re.match(r'^\s*"?answer"?\s*:\s*"<string>"\s*,?\s*$', t, flags=re.I):
            return True
        # Allgemeine Platzhalter
        if _is_placeholder(t):
            return True
        return False

    if last_idx >= 0:
        parts: List[str] = []

        # 1) Inhalt der Answer:-Zeile (kann schon ein Wort enthalten, z.B. "The")
        if last_first and not _is_placeholder(last_first):
            parts.append(last_first)

        # 2) Folgezeilen einsammeln, JSON/Noise überspringen, bis echter Blockwechsel
        j = last_idx + 1
        while j < len(lines):
            nxt = lines[j]
            if _is_stop_marker(nxt):
                break
            if not _is_noise_line(nxt):
                parts.append(nxt.strip())
            j += 1

        # 3) Zusammenfügen + Cleanup
        cand_raw = " ".join(parts)
        # Sonderfall: Wenn erster Split nach "Answer:" ein abgeschnittenes Wort war (z.B. nur "The"),
        # sammeln wir trotzdem weiter – das haben wir oben bereits durch das while erledigt.
        cand = _cleanup(cand_raw)

        # 4) Guardrails: minimale „Substanz“ fordern (z.B. mind. 2 Tokens ODER >= 8 Zeichen)
        if cand and not _is_placeholder(cand):
            tokens = cand.split()
            if len(tokens) >= 2 or len(cand) >= 8:
                return {"answer": cand}

    # 1) JSON-Objekte scannen
    for m in re.findall(r"\{[^{}]*\}", s, flags=re.DOTALL):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                for k in ["answer", "final_answer", "predicted_answer", "output", "result", "antwort", "loesung", "lösung"]:
                    if k in obj and str(obj[k]).strip():
                        val = _cleanup(str(obj[k]).strip())
                        if val and not _is_placeholder(val):
                            return {"answer": val}
        except Exception:
            pass

    # 2) Einzeilige Key-Value-Patterns
    kv_patterns = [
        r"(?i)\bfinal\s*answer\s*[:=]\s*(.+)",
        r"(?i)\banswer\s*[:=]\s*(.+)",
        r"(?i)\bthe\s+answer\s+is\s*(.+)",
        r"(?i)\bantwort\s*[:=]\s*(.+)",
        r"(?i)\blösung\s*[:=]\s*(.+)",
        r"(?i)\bloesung\s*[:=]\s*(.+)",
        r"(?i)\bergebnis\s*[:=]\s*(.+)",
        r"(?i)\bselected\s*option\s*[:=]\s*(.+)",
        r"(?i)\bfinal\s*[:=]\s*(.+)",
    ]
    for pat in kv_patterns:
        m = re.search(pat, s)
        if m:
            cand = _cleanup(m.group(1))
            if cand and not _is_placeholder(cand):
                return {"answer": cand}

    # 3) Längste sinnvolle Zeile als letzter Fallback
    best_line = None
    for line in lines:
        t = line.strip()
        if t and not _is_placeholder(t):
            if best_line is None or len(t) > len(best_line):
                best_line = t
    if best_line:
        return {"answer": _cleanup(best_line)}

    return {"answer": None}


# =====================
# Parsers per task (no confidence)
# =====================

def parse_boolq_output(output: Any) -> Dict[str, Any]:
    ans = universal_answer_parser(output).get("answer")
    return {"answer": normalize_bool_answer(ans) if ans is not None else None}


def parse_trivia_output(output: Any) -> Dict[str, Any]:
    ans = universal_answer_parser(output).get("answer")
    return {"answer": normalize_text_answer(ans) if ans else None}

# Question-aware Variante (für Dumps mit vielen Question:/Answer:-Blöcken)
from difflib import SequenceMatcher as _SM

def parse_trivia_output_qaware(output: Any, question: str, *, min_sim: float = 0.55) -> Dict[str, Any]:
    text = "" if output is None else str(output)
    qn = normalize_text_answer(question or "")
    pattern = re.compile(r"(?is)question\s*:\s*(.*?)\s*answer\s*[:=]\s*(.*?)(?=(?:\n\s*question\s*:)|\Z)")
    best_ans, best_score = None, -1.0
    for q, a in pattern.findall(text):
        q_norm = normalize_text_answer(q)
        a = (a or "").strip()
        if not q_norm or not a:
            continue
        first = a.splitlines()[0].strip().strip('"').strip("'")
        if not first or _is_placeholder(first):
            continue
        sim = _SM(None, q_norm, qn).ratio()
        if sim > best_score:
            best_score, best_ans = sim, first
    if best_ans and best_score >= min_sim:
        return {"answer": normalize_text_answer(best_ans)}
    return parse_trivia_output(output)


def parse_squad_output(output: Any) -> Dict[str, Any]:
    ans = universal_answer_parser(output).get("answer")
    return {"answer": normalize_text_answer(ans) if ans else None}


# =====================
# Row-wise evaluation helpers (exact + fuzzy)
# =====================

def _token_set_ratio(a: str, b: str) -> float:
    ta = set((a or "").split())
    tb = set((b or "").split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


def _fuzzy_match(pred: str, gold: str, threshold: float = 0.8) -> bool:
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    if (len(pred) >= 3 and pred in gold) or (len(gold) >= 3 and gold in pred):
        return True
    if SequenceMatcher(None, pred, gold).ratio() >= threshold:
        return True
    if _token_set_ratio(pred, gold) >= threshold:
        return True
    return False


def evaluate_trivia_answer(predicted: Optional[str], gold_answers: List[str], threshold: float = 0.8) -> bool:
    if not predicted or not gold_answers:
        return False
    p = normalize_text_answer(predicted)
    if not p:
        return False
    for g in gold_answers:
        g_norm = normalize_text_answer(g)
        if not g_norm:
            continue
        if p == g_norm:
            return True
        if _fuzzy_match(p, g_norm, threshold=threshold):
            return True
    return False


def evaluate_squad_answer(predicted: Optional[str], gold_answers: List[str], threshold: float = 0.8) -> bool:
    return evaluate_trivia_answer(predicted, gold_answers, threshold)


# =====================
# Row-wise parse+evaluate pipelines (no confidence)
# =====================

def parse_and_evaluate_trivia(df: pd.DataFrame, *, question_aware: bool = False) -> pd.DataFrame:
    df = df.copy()
    if "parsed_answer" not in df.columns:
        if question_aware and "question" in df.columns:
            parsed = [parse_trivia_output_qaware(mo, q) for mo, q in zip(df["model_output"], df["question"]) ]
        else:
            parsed = df["model_output"].apply(parse_trivia_output)
        df["parsed_answer"] = pd.Series(parsed).apply(lambda x: x.get("answer"))
    if "answers" in df.columns:
        gold_list = [safe_parse_answers(g) for g in df["answers"]]
        df["is_correct"] = [
            evaluate_trivia_answer(p, g)
            for p, g in zip(df["parsed_answer"], gold_list)
        ]
    return df


def parse_and_evaluate_squad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "parsed_answer" not in df.columns:
        parsed = df["model_output"].apply(parse_squad_output)
        df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    if "answers" in df.columns:
        df["is_correct"] = [
            evaluate_squad_answer(p, safe_parse_answers(g))
            for p, g in zip(df["parsed_answer"], df["answers"])
        ]
    return df


# =====================
# Majority-vote aggregation (no confidence)
# =====================

def _majority_vote_norm(series: pd.Series, normalizer) -> Tuple[Optional[str], int]:
    normed = series.map(lambda x: normalizer(x) if pd.notna(x) else None)
    mask = normed.notna()
    normed = normed[mask]
    if len(normed) == 0:
        return None, 0
    counts = normed.value_counts()
    top_norm = counts.idxmax()
    originals = series[mask]
    rep_candidates = originals[normed == top_norm]
    rep_answer = rep_candidates.iloc[0] if len(rep_candidates) else top_norm
    return rep_answer, counts.size


def aggregate_answers_trivia(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    def agg(group):
        rep, k = _majority_vote_norm(group['parsed_answer'], normalize_text_answer)
        return pd.Series({'agg_answer': rep, 'answer_variants': k})
    return df.groupby(group_col, as_index=False).apply(agg).reset_index(drop=True)


def aggregate_answers_squad(df: pd.DataFrame, group_col: str = "question_id") -> pd.DataFrame:
    return aggregate_answers_trivia(df, group_col)


# =====================
# Evaluation of aggregated predictions (no confidence)
# =====================

def evaluate_trivia_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, group_col: str = "question_id") -> Tuple[pd.DataFrame, float]:
    gold = df_full.drop_duplicates(group_col)[[group_col, "answers"]].copy()
    merged = df_agg.merge(gold, on=group_col, how="left")
    merged["is_correct"] = [
        evaluate_trivia_answer(p, safe_parse_answers(g))
        for p, g in zip(merged["agg_answer"], merged["answers"])
    ]
    acc = float(merged["is_correct"].mean()) if len(merged) else 0.0
    return merged, acc


def evaluate_squad_aggregated(df_agg: pd.DataFrame, df_full: pd.DataFrame, group_col: str = "question_id") -> Tuple[pd.DataFrame, float]:
    return evaluate_trivia_aggregated(df_agg, df_full, group_col)


# =====================
# End-to-end pipelines (parse → aggregate → evaluate)
# =====================

def parse_aggregate_evaluate_trivia_multi(df: pd.DataFrame, group_col: str = "question_id", *, question_aware: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    df_parsed = parse_and_evaluate_trivia(df, question_aware=question_aware)
    df_agg = aggregate_answers_trivia(df_parsed, group_col)
    eval_df, acc = evaluate_trivia_aggregated(df_agg, df_parsed, group_col)
    return df_parsed, df_agg, eval_df, acc


def parse_aggregate_evaluate_squad_multi(df: pd.DataFrame, group_col: str = "question_id", remove_unanswerable: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    df_parsed = parse_and_evaluate_squad(df)
    if remove_unanswerable and "is_impossible" in df_parsed.columns:
        df_parsed = df_parsed[~df_parsed["is_impossible"].fillna(False)].copy()
    df_agg = aggregate_answers_squad(df_parsed, group_col)
    eval_df, acc = evaluate_squad_aggregated(df_agg, df_parsed, group_col)
    return df_parsed, df_agg, eval_df, acc