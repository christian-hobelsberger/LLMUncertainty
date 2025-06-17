from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import Counter
from typing import List, Dict
import pandas as pd
import os
import json
from pathlib import Path
import re


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



def build_prompt(row, include_context: bool) -> str:
    role_init = (
        "You are an expert QA assistant. Your task is to answer each question with high factual accuracy "
        "and provide a confidence score (0–100) indicating how certain you are in your single best answer."
    )

    expectation = (
        "Respond in a single-line JSON object with exactly two fields:\n"
        "{\"answer\": <string>, \"confidence\": <integer from 0 to 100>}.\n"
        "Only provide one answer. Do not list alternatives. No markdown, no prose, no code blocks."
    )

    # Few-shot examples
    examples = (
        "Example 1:\n"
        "Question: What is the capital of France?\n"
        "Answer: {\"answer\": \"Paris\", \"confidence\": 95}\n\n"
        "Example 2:\n"
        "Question: Who painted the Mona Lisa?\n"
        "Answer: {\"answer\": \"Leonardo da Vinci\", \"confidence\": 98}\n\n"
    )

    final_instruction = (
        "Now answer the following:\n"
    )

    if include_context:
        return (
            f"{role_init}\n\n"
            f"{expectation}\n\n"
            f"{examples}"
            f"{final_instruction}"
            f"Context: {row['context']}\n"
            f"Question: {row['question']}"
        )
    else:
        return (
            f"{role_init}\n\n"
            f"{expectation}\n\n"
            f"{examples}"
            f"{final_instruction}"
            f"Question: {row['question']}"
        )


# ── Main experiment runner ─────────────────────────────────────────────────────
def run_verbalized_confidence_experiment(
    model_wrapper,
    n_samples: int,
    dataset_name: str,
    output_ending: str
):
    dataset_name = dataset_name.lower()
    if dataset_name == "squad":
        df = pd.read_csv("data/SQuAD2_merged.csv")
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        df["prompt"] = df.apply(lambda row: build_prompt(row, include_context=True), axis=1)
    elif dataset_name == "trivia":
        df = pd.read_csv("data/trivia_all.csv")
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        df["prompt"] = df.apply(lambda row: build_prompt(row, include_context=False), axis=1)
    else:
        raise ValueError("dataset_name must be either 'squad' or 'trivia'.")

    # Generate model outputs
    print("Generating responses...")
    df["model_output"] = df["prompt"].apply(lambda p: model_wrapper.prompt(p))

    # Parse JSON response
    parsed = df["model_output"].apply(clean_and_parse_json)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")

    # Save output
    Path("output").mkdir(parents=True, exist_ok=True)
    output_file = f"output/{dataset_name}_verbalized_confidence{output_ending}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Define all model wrappers and names for tracking
    models = {
        "llama": load_llama(),
        "qwen": load_qwen(),
        "gemma": load_gemma(),
    }

    datasets = ["squad", "trivia"]

    for model_name, model in models.items():
        for dataset in datasets:
            print(f"🔍 Running {model_name} on {dataset}...")
            run_verbalized_confidence_experiment(
                model_wrapper=model,
                n_samples=7,
                dataset_name=dataset,
                output_ending=f"_fewshot_json_clean_v5_{model_name}"
            )



    """ llama = load_llama()  # assumed external model wrapper
    run_verbalized_confidence_experiment(
        model_wrapper=llama,
        n_samples=10,
        dataset_name="squad",  # or "trivia"
        output_ending="_fewshot_json_clean_v3"
    ) """

    """ llama = load_llama()
    print("LLaMA sample:\nQ: ", llama.prompt("What is the capital of France? Include confidence as a percentage."))

    qwen = load_qwen()
    print("Qwen sample:\nQ:", qwen.prompt("Who painted the Mona Lisa? Include confidence as a percentage."))

    gemma = load_gemma()
    print("Gemma sample:\nQ:", gemma.prompt("What is the boiling point of water in Celsius? Include confidence as a percentage.")) """