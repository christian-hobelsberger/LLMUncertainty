# ------------------------- CoCoA single-pass runner (BoolQ / Trivia / SQuAD) -------------------------

import os
import re
import ast
import json
import time
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig  

from lm_polygraph import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import CocoaMSP, MaximumSequenceProbability
from lm_polygraph.utils.generation_parameters import GenerationParameters


# ------------------------- MODEL WRAPPER -------------------------

class LLMWrapper:
    def __init__(self, model_id: str, trust_remote_code: bool = False, is_batchable: bool = True):
        self.model_id = model_id
        self.is_batchable = is_batchable
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=trust_remote_code
        )
        self.model.eval()

def load_llama():
    return LLMWrapper("meta-llama/Llama-3.2-3B-Instruct")

def load_qwen():
    return LLMWrapper("Qwen/Qwen3-8B", trust_remote_code=True)

def load_gemma():
    return LLMWrapper("google/gemma-3-1b-it")

def load_phi2():
    return LLMWrapper("microsoft/phi-2", trust_remote_code=True)


# ------------------------- DATASETS -------------------------

def load_dataset(dataset_name: str, sample_size: int) -> pd.DataFrame:
    """
    Adjust paths to your local files if needed.
    Expected minimal columns:
      - boolq: question, passage, answer (True/False or yes/no)
      - trivia: question, answers (stringified list or list)
      - squad: context, question, answers (stringified list or list), is_impossible (optional), id or question_id
    """
    dataset_paths = {
        "squad":  "data/SQuAD2_merged.csv",
        "trivia": "data/trivia_all.csv",
        "boolq":  "data/boolQ.csv",
        "gsm8k":  "data/gsm8k_cleaned.csv",
    }
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    df = pd.read_csv(dataset_paths[dataset_name])

    # normalize IDs if present
    if "id" in df.columns:
        df["question_id"] = df["id"]
    elif "question_id" not in df.columns:
        df["question_id"] = np.arange(len(df))

    # ensure SQuAD is_impossible exists and boolean
    if dataset_name == "squad":
        if "is_impossible" not in df.columns:
            df["is_impossible"] = False
        else:
            if df["is_impossible"].dtype != bool:
                df["is_impossible"] = df["is_impossible"].apply(lambda v: str(v).strip().lower() in {"true","1","yes"})

    # downsample
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


# ------------------------- PROMPTS -------------------------

def build_input_text(row, dataset_name: str) -> str:
    """
    Prompts tuned to yield short, extractive answers that your parser can handle later.
    """
    if dataset_name == "squad":
        return (
            "You are a question-answering assistant. Answer with a short span from the context only.\n"
            f"Context: {row.get('context','')}\n"
            f"Question: {row.get('question','')}\n"
            "Answer:"
        )
    elif dataset_name == "boolq":
        # encourage single word yes/no; parser expects boolean/yes/no mapping
        return (
            "You are a reading comprehension assistant. Answer with exactly one word: 'yes' or 'no'.\n"
            f"Passage: {row.get('passage','')}\n"
            f"Question: {row.get('question','')}\n"
            "Answer:"
        )
    elif dataset_name == "trivia":
        return (
            "Answer the trivia question concisely with the best short answer.\n"
            f"Question: {row.get('question','')}\n"
            "Answer:"
        )
    elif dataset_name == "gsm8k":
        return (
            "You are a math problem solver. Solve the following math problem step-by-step. Answer with the exact numerical answer, which is an integer.\n"
            f"Question: {row.get('question','')}\n"
            "Answer:"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# ------------------------- UTILITIES -------------------------

def normalize_bool_label(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    s = str(x).strip().lower()
    if s in {"true", "yes", "1"}:
        return "true"
    if s in {"false", "no", "0"}:
        return "false"
    return s

def safe_answers_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [val]
    return [] if pd.isna(val) else [val]

def to_model_output_json(answer_str: str, unc_score: float) -> str:
    """
    Emit JSON-ish string to be compatible with your universal parser:
      {"answer": "<text>", "uncertainty_score": 0.8246}
    """
    # escape internal quotes in answer
    safe_ans = str(answer_str).replace('"', "'")
    unc_float = float(unc_score) if np.isfinite(unc_score) else None
    return json.dumps({"answer": safe_ans, "uncertainty_score": unc_float}, ensure_ascii=False)


# ------------------------- MAIN EXPERIMENT -------------------------

def run_polygraph_inference(
    model_wrapper: LLMWrapper,
    ue_method_class,
    dataset_name: str,
    sample_size: int,
    output_dir: str
):
    model_id = model_wrapper.model_id
    model_name_safe = model_id.split("/")[-1].replace("-", "_").lower()
    tokenizer = model_wrapper.tokenizer
    raw_model = model_wrapper.model
    gen_params = GenerationParameters(
        do_sample=False,
        max_new_tokens=750,
        )
    polygraph_model = WhiteboxModel(raw_model, tokenizer, generation_parameters=gen_params)

    ue_method = ue_method_class()

    print(f"\n📊 Loading {dataset_name} dataset with {sample_size} samples...")
    df = load_dataset(dataset_name, sample_size).copy()
    df["input_text"] = df.apply(lambda row: build_input_text(row, dataset_name), axis=1)

    print(f"🧠 [{model_name_safe}] Estimating uncertainty with {ue_method_class.__name__}...")
    preds, uncs, outputs = [], [], []

    for text in tqdm(df["input_text"].tolist(), desc=f"{model_name_safe} on {dataset_name}"):
        out = estimate_uncertainty(polygraph_model, ue_method, text)
        pred = (out.generation_text or "").strip()
        unc  = float(out.uncertainty) if out.uncertainty is not None else np.nan

        preds.append(pred)
        uncs.append(unc)
        outputs.append(to_model_output_json(pred, unc))

    # attach predictions
    df["model_answer"] = preds
    df["uncertainty_score"] = uncs
    df["model_output"] = outputs  # parser-friendly string

    # gold labels (so you can evaluate later with your fuzzy matchers)
    if dataset_name == "squad":
        # prefer 'answers' list if available; otherwise 'answer_text'
        if "answers" in df.columns:
            df["answers"] = df["answers"].apply(safe_answers_list)
        elif "answer_text" in df.columns:
            df["answers"] = df["answer_text"].apply(lambda x: [x] if pd.notna(x) else [])
        # is_impossible ensured in loader; keep for later filtering
    elif dataset_name == "trivia":
        if "answers" in df.columns:
            df["answers"] = df["answers"].apply(safe_answers_list)
        elif "answer" in df.columns:
            df["answers"] = df["answer"].apply(lambda x: [x] if pd.notna(x) else [])
    elif dataset_name == "boolq":
        if "answer" in df.columns:
            df["answer"] = df["answer"].apply(normalize_bool_label)

    # metadata
    df["dataset"] = dataset_name
    df["model_id"] = model_id
    df["ue_method"] = ue_method_class.__name__
    df["schema_version"] = "msp" # ["cocoav1"]
    df["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # write
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ofile = Path(output_dir) / f"{dataset_name}_{model_name_safe}_{ue_method_class.__name__}_{len(df)}.csv"
    df.to_csv(ofile, index=False)
    print(f"✅ Saved results to: {ofile}")


# ------------------------- ENTRYPOINT -------------------------

def main():
    # choose models
    models = {
        "llama": load_llama,
        # "qwen":  load_qwen,
        # "phi2":  load_phi2,
        # "gemma": load_gemma,
    }

    # datasets you want to run
    datasets = ["trivia"] # ["boolq", "squad", "trivia", "gsm8k"]
    sample_size = 500

    ue_method_class = MaximumSequenceProbability # CocoaMSP
    output_dir = "output/polygraph/MSP/final/" # output/polygraph/CocoaMSP/final/

    for name, loader in models.items():
        print(f"\n🚀 Running model: {name}")
        wrapper = loader()

        for dataset in datasets:
            run_polygraph_inference(
                model_wrapper=wrapper,
                ue_method_class=ue_method_class,
                dataset_name=dataset,
                sample_size=sample_size,
                output_dir=output_dir
            )

        # clean up GPU memory
        del wrapper
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    main()
