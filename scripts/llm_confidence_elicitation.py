from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import Counter
from typing import List, Dict
import pandas as pd
import os
import json
from pathlib import Path
import re
import torch
import gc


class LLMWrapper:
    def __init__(self, model_id: str, trust_remote_code: bool = False):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=trust_remote_code)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False  # Ensures only generated output is returned
        )

    def prompt(self, text: str, max_new_tokens: int = 650, temperature: float = 0.7) -> str:
        output = self.pipe(text, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        return output[0]['generated_text'].strip()

    def batch_prompt(self, prompts: List[str], max_new_tokens: int = 650, temperature: float = 0.7) -> List[str]:
        return [self.prompt(prompt, max_new_tokens, temperature) for prompt in prompts]

    def sample_variants(self, prompt: str, n: int = 5, max_new_tokens: int = 650, temperature: float = 0.7) -> Dict[str, int]:
        responses = [self.prompt(prompt, max_new_tokens, temperature) for _ in range(n)]
        return Counter(responses)


# Predefined model wrappers
def load_llama():
    return LLMWrapper("meta-llama/Llama-3.2-1B-Instruct")

def load_qwen():
    return LLMWrapper("Qwen/Qwen3-8B", trust_remote_code=True)

def load_gemma():
    return LLMWrapper("google/gemma-3-1b-it")

def load_cogito():
    return LLMWrapper("deepcogito/cogito-v1-preview-qwen-14B", trust_remote_code=True)

# def load_deepseek_r1_0528():
#     return LLMWrapper("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)

def load_phi2():
    return LLMWrapper("microsoft/phi-2", trust_remote_code=True)

# Main experiment function
# ── Clean and strictly parse expected JSON output ──────────────────────────────
def clean_and_parse_json(output):
    """
    Extracts the first valid JSON object with:
    - both 'answer' and 'confidence', OR
    - just 'answer' (fallback if confidence is missing)

    Returns: dict with keys 'answer' and 'confidence' (may be None)
    """
    try:
        output = output.strip()

        # Find all JSON-like blocks in the string
        matches = re.findall(r'\{.*?\}', output, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    if "answer" in parsed and "confidence" in parsed:
                        return parsed
                    elif "answer" in parsed:
                        return {"answer": parsed["answer"], "confidence": None}
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return {"answer": None, "confidence": None}



def build_prompt(row, include_context: bool, dataset_name: str = "") -> str:
    if dataset_name == "gsm8k":
        return (
            "You are a math tutor. Solve the following problem carefully and provide a numerical answer.\n\n"
            "Respond in the following format:\n"
            "{\"answer\": <string>, \"confidence\": <integer from 0 to 100>}\n"
            f"Question: {row['question']}\n\n"
        )

    elif dataset_name == "boolq":
        return (
            "You are a reading comprehension assistant. Read the following passage and determine whether the answer "
            "to the question is 'yes' or 'no'. Also include a confidence score (0–100).\n\n"
            "Respond in JSON format with exactly two fields:\n"
            "{\"answer\": <yes/no>, \"confidence\": <integer from 0 to 100>}\n\n"
            f"Passage: {row['passage']}\n"
            f"Question: {row['question']}"
        )

    # Default prompt style for squad, trivia
    role_init = (
        "You are an expert QA assistant. Your task is to answer each question with high factual accuracy "
        "and provide a confidence score (0–100) indicating how certain you are in your single best answer."
    )
    expectation = (
        "Respond in a single-line JSON object with exactly two fields:\n"
        "{\"answer\": <string>, \"confidence\": <integer from 0 to 100>}.\n"
        "Only provide one answer. Do not list alternatives."
    )
    final_instruction = "Now answer the following:\n"

    if include_context:
        context = row["context"] if dataset_name == "squad" else row["passage"]
        return (
            f"{role_init}\n\n{expectation}\n\n{final_instruction}"
            f"Context: {context}\n"
            f"Question: {row['question']}"
        )
    else:
        return (
            f"{role_init}\n\n{expectation}\n\n{final_instruction}"
            f"Question: {row['question']}"
        )

        

# ── Main experiment runner ─────────────────────────────────────────────────────
def run_verbalized_confidence_experiment(
    model_wrapper,
    n_samples: int,
    dataset_name: str,
    folder_name: str,
    output_ending: str
):
    dataset_name = dataset_name.lower()
    
    if dataset_name == "squad":
        df = pd.read_csv("data/SQuAD2_merged.csv")
        include_context = True
    elif dataset_name == "trivia":
        df = pd.read_csv("data/trivia_all.csv")
        include_context = False
    elif dataset_name == "gsm8k":
        df = pd.read_csv("data/gsm8k_cleaned.csv")
        include_context = False
    elif dataset_name == "boolq":
        df = pd.read_csv("data/boolQ.csv")
        include_context = True
    else:
        raise ValueError("dataset_name must be one of: 'squad', 'trivia', 'gsm8k', 'boolq'")

    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    df["prompt"] = df.apply(lambda row: build_prompt(row, include_context, dataset_name), axis=1)

    print("Generating responses...")
    df["model_output"] = df["prompt"].apply(lambda p: model_wrapper.prompt(p))

    parsed = df["model_output"].apply(clean_and_parse_json)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")

    Path("output").mkdir(parents=True, exist_ok=True)
    output_file = f"output/{folder_name}{dataset_name}_verbalized_confidence{output_ending}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")



# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Define all model wrappers and names for tracking
    models = {
    "llama": load_llama,
    "qwen": load_qwen,
    "gemma": load_gemma,
    "cogito": load_cogito,
    "phi2": load_phi2
    }
    datasets = ["squad", "trivia", "gsm8k", "boolq"]
    folder_name = "llm_confidence_elicitation/batch_2_test/"

    Path(f"output/{folder_name}").mkdir(parents=True, exist_ok=True)

for model_name, model_loader in models.items():
    print(f"\n🚀 Loading model: {model_name}")
    model = model_loader()

    for dataset in datasets:
        print(f"🔍 Running {model_name} on {dataset}...")
        run_verbalized_confidence_experiment(
            model_wrapper=model,
            n_samples=2,
            dataset_name=dataset,
            folder_name=folder_name,
            output_ending=f"_{model_name}_batch_2_test"
        )

    # 🔻 Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
