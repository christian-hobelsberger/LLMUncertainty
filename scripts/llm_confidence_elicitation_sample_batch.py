import os
import re
import json
import gc
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
#                                LLM Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class LLMWrapper:
    def __init__(self, model_id: str, trust_remote_code: bool = False):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=trust_remote_code)
        self.model.eval()

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_samples(self, prompts: List[str], n: int = 5, max_new_tokens: int = 300,
                         temperature: float = 0.7, separate: bool = True) -> List[List[str]]:
        all_outputs = []

        if separate:
            for prompt in tqdm(prompts, desc="Generating separately"):
                single_outputs = []
                for _ in range(n):
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    generated = out[:, inputs["input_ids"].shape[1]:]
                    decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
                    single_outputs.append(decoded)
                all_outputs.append(single_outputs)
        else:
            for prompt in tqdm(prompts, desc="Generating with top-k sampling"):
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        num_return_sequences=n,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                generated = out[:, inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                all_outputs.append([text.strip() for text in decoded])

        return all_outputs


# ─────────────────────────────────────────────────────────────────────────────
#                                Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(row, include_context: bool, dataset_name: str = "") -> str:
    if dataset_name == "gsm8k":
        return (
            "You are a math tutor. Solve the following problem carefully and provide a numerical answer.\n\n"
            "{\"answer\": \"<string>\", \"confidence\": <integer from 0 to 100>}\n"
            f"Question: {row['question']}"
        )
    elif dataset_name == "boolq":
        return (
            "You are a reading comprehension assistant. Given a passage and a yes/no question, respond with one JSON object.\n\n"
            "{\"answer\": \"True\" or \"False\", \"confidence\": integer from 0 to 100}\n"
            f"Passage: {row['passage']}\nQuestion: {row['question']}"
        )
    elif dataset_name == "trivia":
        return (
            "You are a fact-checking assistant. Answer the following trivia question and provide a confidence score (0–100).\n\n"
            "{\"answer\": \"<string>\", \"confidence\": <integer from 0 to 100>}\n"
            f"Question: {row['question']}"
        )
    elif dataset_name == "squad":
        return (
            "You are a QA assistant. Use the context to answer the question with a short span.\n\n"
            "{\"answer\": \"<short answer>\", \"confidence\": <integer from 0 to 100>}\n"
            f"Context: {row['context']}\nQuestion: {row['question']}"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
#                                JSON Parser
# ─────────────────────────────────────────────────────────────────────────────
def clean_and_parse_json(output: str) -> dict:
    try:
        output = output.strip()
        matches = re.findall(r'\{.*?\}', output, re.DOTALL)
        for match in matches:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                if "answer" in parsed and "confidence" in parsed:
                    return parsed
                elif "answer" in parsed:
                    return {"answer": parsed["answer"], "confidence": None}
    except Exception:
        pass
    return {"answer": None, "confidence": None}


# ─────────────────────────────────────────────────────────────────────────────
#                            Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_verbalized_confidence_experiment(
    model_wrapper: LLMWrapper,
    dataset_name: str,
    sample_size_per_question: int,
    separate_prompting: bool,
    output_path: str,
    n_samples: int = 100
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
        raise ValueError("Unsupported dataset.")

    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    df["prompt"] = df.apply(lambda row: build_prompt(row, include_context, dataset_name), axis=1)

    outputs_per_question = model_wrapper.generate_samples(
        prompts=df["prompt"].tolist(),
        n=sample_size_per_question,
        separate=separate_prompting
    )

    all_records = []
    for i, prompt_outputs in enumerate(outputs_per_question):
        original_row = df.loc[i].to_dict()
        for j, out in enumerate(prompt_outputs):
            parsed = clean_and_parse_json(out)
            row_with_sample = {
                **original_row,  # Includes all original columns
                "question_id": i,
                "sample_id": j,
                "model_output": out,
                "parsed_answer": parsed.get("answer"),
                "parsed_confidence": parsed.get("confidence")
            }
            all_records.append(row_with_sample)

    result_df = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Saved results to {output_path}")



# ─────────────────────────────────────────────────────────────────────────────
#                                   Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    models = {
        "llama": ("meta-llama/Llama-3.2-1B-Instruct", True),
        # "qwen": ("Qwen/Qwen3-8B", True),
        # "phi2": ("microsoft/phi-2", True),
    }

    datasets = ["boolq"] # ["trivia", "squad", "gsm8k", "boolq"]
    sample_size_per_question = 5
    separate_prompting = False
    base_output_dir = "output/verbalized_confidence_multi_model_full_results"

    for model_name, (model_id, trust_remote_code) in models.items():
        print(f"\n🚀 Loading model: {model_name}")
        model = LLMWrapper(model_id=model_id, trust_remote_code=trust_remote_code)

        for dataset in datasets:
            print(f"🔍 Running {model_name} on {dataset}...")
            output_file = os.path.join(
                base_output_dir,
                f"{dataset}_{model_name}_k{sample_size_per_question}_{'sep' if separate_prompting else 'topk'}.csv"
            )

            run_verbalized_confidence_experiment(
                model_wrapper=model,
                dataset_name=dataset,
                sample_size_per_question=sample_size_per_question,
                separate_prompting=separate_prompting,
                output_path=output_file,
                n_samples=5
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()