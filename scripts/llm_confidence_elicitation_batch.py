from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from collections import Counter
import pandas as pd
import torch
import json
import re
import gc
from pathlib import Path
from tqdm import tqdm


class LLMWrapper:
    def __init__(self, model_id: str, trust_remote_code: bool = False):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        # Fix padding issues
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=trust_remote_code)

    def prompt(self, text: str, max_new_tokens: int = 650, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def batch_prompt(self, prompts: List[str], max_new_tokens: int = 650, temperature: float = 0.7, batch_size: int = 4) -> List[str]:
        outputs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            try:
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                outputs.extend([text.strip() for text in decoded])
            except Exception as e:
                print(f"⚠️ Batching failed: {e}. Falling back to sequential prompting.")
                for prompt in tqdm(batch, desc="Fallback"):
                    outputs.append(self.prompt(prompt, max_new_tokens, temperature))
        return outputs


# Model Loaders
def load_llama():
    return LLMWrapper("meta-llama/Llama-3.2-1B-Instruct")

def load_qwen():
    return LLMWrapper("Qwen/Qwen3-8B", trust_remote_code=True)

def load_gemma():
    return LLMWrapper("google/gemma-3-1b-it")

def load_cogito():
    return LLMWrapper("deepcogito/cogito-v1-preview-qwen-14B", trust_remote_code=True)

def load_phi2():
    return LLMWrapper("microsoft/phi-2", trust_remote_code=True)


def clean_and_parse_json(output):
    try:
        output = output.strip()
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
    role_init = (
        "You are an expert QA assistant. Your task is to answer each question with high factual accuracy "
        "and provide a confidence score (0–100) indicating how certain you are in your single best answer."
    )
    expectation = (
        "Respond in a single-line JSON object with exactly two fields:\n"
        "{\"answer\": <string>, \"confidence\": <integer from 0 to 100>}.\n"
        "Only provide one answer. Do not list alternatives. No markdown, no prose, no code blocks."
    )
    examples = (
        "Example 1:\n"
        "Question: What is the capital of France?\n"
        "Answer: {\"answer\": \"Paris\", \"confidence\": 95}\n\n"
        "Example 2:\n"
        "Question: Who painted the Mona Lisa?\n"
        "Answer: {\"answer\": \"Leonardo da Vinci\", \"confidence\": 98}\n\n"
    )
    final_instruction = "Now answer the following:\n"

    if include_context:
        context = row["context"] if dataset_name == "squad" else row["passage"]
        return (
            f"{role_init}\n\n{expectation}\n\n{examples}"
            f"{final_instruction}Context: {context}\nQuestion: {row['question']}"
        )
    else:
        return (
            f"{role_init}\n\n{expectation}\n\n{examples}"
            f"{final_instruction}Question: {row['question']}"
        )


def run_verbalized_confidence_experiment(
    model_wrapper,
    n_samples: int,
    dataset_name: str,
    folder_name: str,
    output_ending: str,
    batch_size: int = 4
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

    print("🧠 Generating responses...")
    prompts = df["prompt"].tolist()
    df["model_output"] = model_wrapper.batch_prompt(prompts, batch_size=batch_size)

    parsed = df["model_output"].apply(clean_and_parse_json)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")

    Path("output").mkdir(parents=True, exist_ok=True)
    output_file = f"output/{folder_name}{dataset_name}_verbalized_confidence{output_ending}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")


if __name__ == "__main__":
    models = {
        "llama": load_llama,
        "qwen": load_qwen,
        "gemma": load_gemma,
        "cogito": load_cogito,
        "phi2": load_phi2
    }

    datasets = ["squad", "trivia", "gsm8k", "boolq"]
    folder_name = "llm_confidence_elicitation/batch_75/"

    # 🧱 Ensure the output directory exists
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
                output_ending=f"_{model_name}batch_75",
                batch_size=4
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

