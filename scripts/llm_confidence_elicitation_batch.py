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

# Disable TorchDynamo for compatibility
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 256


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

    def prompt(self, text: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            # Check for truncation
            original_tokens = len(self.tokenizer.encode(text))
            if original_tokens > 2048:
                print(f"⚠️ Warning: Individual prompt truncated from {original_tokens} to 2048 tokens")
            
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            # Extract only new tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0, input_length:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return decoded.strip()
        except Exception as e:
            print(f"⚠️ Error in individual prompt: {e}")
            return "?"

    def batch_prompt(self, prompts: List[str], max_new_tokens: int = 300, temperature: float = 0.7, batch_size: int = 4) -> List[str]:
        outputs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            try:
                # Set max_length to prevent silent truncation issues
                max_length = 2048  # Adjust based on your model's context window
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                input_ids = inputs["input_ids"]
                
                # Check for truncation and warn
                for j, prompt in enumerate(batch):
                    original_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
                    if original_tokens > max_length:
                        print(f"⚠️ Warning: Prompt {i+j+1} was truncated from {original_tokens} to {max_length} tokens")
                
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Extract only the new generated tokens
                input_length = input_ids.shape[1]
                generated_tokens = generated[:, input_length:]
                decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                # Validate outputs and fallback if needed
                for j, (prompt, output) in enumerate(zip(batch, decoded)):
                    cleaned = output.strip()
                    if not cleaned or cleaned == "?" or len(cleaned) < 5:
                        print(f"⚠️ Poor output detected for prompt {i+j+1}, retrying individually...")
                        cleaned = self.prompt(prompt, max_new_tokens, temperature)
                    outputs.append(cleaned)
                    
            except Exception as e:
                print(f"⚠️ Batching failed: {e}. Falling back to sequential prompting.")
                for prompt in tqdm(batch, desc="Fallback"):
                    outputs.append(self.prompt(prompt, max_new_tokens, temperature))
        return outputs


# Model Loaders
def load_llama():
    return LLMWrapper("meta-llama/Llama-3.2-3B-Instruct")

def load_qwen():
    return LLMWrapper("Qwen/Qwen3-8B", trust_remote_code=True)

def load_gemma():
    return LLMWrapper("google/gemma-3-1b-it")

def load_cogito():
    return LLMWrapper("deepcogito/cogito-v1-preview-qwen-14B", trust_remote_code=True, is_batchable=False)

def load_phi2():
    return LLMWrapper("microsoft/phi-2", trust_remote_code=True)


def clean_and_parse_json(output):
    try:
        output = output.strip()
        
        # Handle empty or single character outputs
        if not output or len(output) <= 1:
            return {"answer": None, "confidence": None, "needs_reprompt": True}
        
        # Try to find JSON objects in the output
        matches = re.findall(r'\{.*?\}', output, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    if "answer" in parsed and "confidence" in parsed:
                        return {"answer": parsed["answer"], "confidence": parsed["confidence"], "needs_reprompt": False}
                    elif "answer" in parsed:
                        return {"answer": parsed["answer"], "confidence": None, "needs_reprompt": False}
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to extract from common patterns
        # Look for "True"/"False" patterns
        true_pattern = r'\b(?:true|True|TRUE)\b'
        false_pattern = r'\b(?:false|False|FALSE)\b'
        
        if re.search(true_pattern, output):
            return {"answer": "True", "confidence": None, "needs_reprompt": True}
        elif re.search(false_pattern, output):
            return {"answer": "False", "confidence": None, "needs_reprompt": True}
            
    except Exception:
        pass
    return {"answer": None, "confidence": None, "needs_reprompt": True}


def build_reprompt_for_json(original_output: str, dataset_name: str = "") -> str:
    """Build a prompt to convert non-JSON output to proper JSON format"""
    base_prompt = (
        "The following text contains an answer but is not in the required JSON format. "
        "Please convert it to the exact JSON format requested:\n\n"
        f"Original response: {original_output}\n\n"
        "Convert this to JSON format with exactly these fields:\n"
        "{\"answer\": \"<the answer from above>\", \"confidence\": <integer from 0 to 100>}\n\n"
    )
    
    if dataset_name == "boolq":
        base_prompt += "For the answer field, use only \"True\" or \"False\".\n"
    elif dataset_name == "gsm8k":
        base_prompt += "For the answer field, provide only the numerical answer.\n"
    
    base_prompt += "Respond ONLY with the JSON object, no other text:"
    
    return base_prompt


def build_prompt(row, include_context: bool, dataset_name: str = "") -> str:
    if dataset_name == "gsm8k":
        return (
            "You are a math tutor. Solve the following problem carefully and provide only the numerical answer and the confidence of the answer.\n\n"
            "Respond in the following format:\n"
            "{\"answer\": \"<string>,\" \"confidence\": <integer from 0 to 100>}\n"
            f"Question: {row['question']}\n\n"
        )

    elif dataset_name == "boolq":
        return (
        "You are a reading comprehension assistant. Given a passage and a yes/no question, respond only with one JSON object containing the answer and the confidence of the answer.\n\n"
        "Format strictly:\n"
        "{\"answer\": \"True\" OR \"False\", \"confidence\": integer from 0 to 100}\n\n"
        "No explanation. No extra text. Only JSON.\n\n"
        f"Passage: {row['passage']}\n"
        f"Question: {row['question']}"
        )
    elif dataset_name == "trivia":
        return (
            "You are a fact-checking assistant. Answer the following trivia question concisely and provide your confidence as a number from 0 to 100.\n\n"
            "Respond in the following JSON format:\n"
            "{\"answer\": \"<string>\", \"confidence\": <integer from 0 to 100>}\n\n"
            "No explanation. No prose. Only JSON.\n\n"
            f"Question: {row['question']}"
        )
    elif dataset_name == "squad":
        return (
            "You are a question answering assistant. Use the context below to answer the question as accurately as possible.\n\n"
            "Respond in the following format:\n"
            "{\"answer\": \"<short answer from the context>\", \"confidence\": <integer from 0 to 100>}\n\n"
            "Answer concisely. Only output JSON — no explanation or extra text.\n\n"
            f"Context: {row['context']}\n"
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


# Previous function definition for run_verbalized_confidence_experiment
""" def run_verbalized_confidence_experiment(
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

    if getattr(model_wrapper, "is_batchable", True):
        df["model_output"] = model_wrapper.batch_prompt(prompts, batch_size=batch_size)
    else:
        df["model_output"] = [model_wrapper.prompt(p) for p in tqdm(prompts, desc=f"{dataset_name} (no batching)")]

    # Check for failed outputs and retry if needed
    failed_indices = []
    for idx, output in enumerate(df["model_output"]):
        if not output or output.strip() == "?" or len(output.strip()) < 3:
            failed_indices.append(idx)
    
    if failed_indices:
        print(f"⚠️ Found {len(failed_indices)} failed outputs, retrying individually...")
        for idx in tqdm(failed_indices, desc="Retrying failed prompts"):
            retry_output = model_wrapper.prompt(prompts[idx])
            if retry_output and retry_output.strip() != "?" and len(retry_output.strip()) >= 3:
                df.loc[idx, "model_output"] = retry_output
            else:
                print(f"⚠️ Prompt {idx} failed even after retry. Prompt length: {len(prompts[idx])}")

    # Parse JSON and identify responses that need reprompting
    parsed = df["model_output"].apply(clean_and_parse_json)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
    
    # Find indices that need reprompting for JSON formatting
    reprompt_indices = []
    for idx, result in enumerate(parsed):
        if result.get("needs_reprompt", False) and df.loc[idx, "model_output"] and df.loc[idx, "model_output"].strip():
            reprompt_indices.append(idx)
    
    if reprompt_indices:
        print(f"🔄 Found {len(reprompt_indices)} responses that need JSON reprompting...")
        for idx in tqdm(reprompt_indices, desc="Reprompting for JSON format"):
            original_output = df.loc[idx, "model_output"]
            reprompt = build_reprompt_for_json(original_output, dataset_name)
            
            # Try reprompting for proper JSON format
            json_output = model_wrapper.prompt(reprompt, max_new_tokens=150, temperature=0.3)
            
            # Parse the reprompted output
            if json_output and json_output.strip():
                reparsed = clean_and_parse_json(json_output)
                if not reparsed.get("needs_reprompt", True):  # Successfully parsed
                    df.loc[idx, "model_output"] = json_output
                    df.loc[idx, "parsed_answer"] = reparsed.get("answer")
                    df.loc[idx, "parsed_confidence"] = reparsed.get("confidence")
                    continue
            
            # If reprompting failed, try to extract basic answer from original
            original_parsed = clean_and_parse_json(original_output)
            if original_parsed.get("answer") is not None:
                df.loc[idx, "parsed_answer"] = original_parsed.get("answer")
                # Set a default confidence if we found an answer but no confidence
                if df.loc[idx, "parsed_confidence"] is None or pd.isna(df.loc[idx, "parsed_confidence"]):
                    df.loc[idx, "parsed_confidence"] = 50  # Default moderate confidence
                print(f"⚠️ Used fallback parsing for prompt {idx}")
            else:
                print(f"⚠️ Could not extract answer from prompt {idx} even after reprompting")

    # Sanitize model ID for use in filename
    safe_model_id = model_wrapper.model_id.replace("/", "_").replace(" ", "_")
    Path(f"output/{folder_name}").mkdir(parents=True, exist_ok=True)
    output_file = f"output/{folder_name}{dataset_name}_verbalized_confidence_{safe_model_id}{output_ending}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}") """



def run_verbalized_confidence_experiment(
    model_wrapper,
    n_samples: int,
    dataset_name: str,
    folder_name: str,
    output_ending: str,
    batch_size: int = 4,
    temperature: float = 0.7
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

    print(f"🧠 Generating responses at T={temperature} ...")
    prompts = df["prompt"].tolist()

    if getattr(model_wrapper, "is_batchable", True):
        df["model_output"] = model_wrapper.batch_prompt(
            prompts, batch_size=batch_size, temperature=temperature
        )
    else:
        df["model_output"] = [
            model_wrapper.prompt(p, temperature=temperature)
            for p in tqdm(prompts, desc=f"{dataset_name} (no batching)")
        ]

    # Check for failed outputs and retry if needed
    failed_indices = []
    for idx, output in enumerate(df["model_output"]):
        if not output or output.strip() == "?" or len(output.strip()) < 3:
            failed_indices.append(idx)
    
    if failed_indices:
        print(f"⚠️ Found {len(failed_indices)} failed outputs, retrying individually...")
        for idx in tqdm(failed_indices, desc="Retrying failed prompts"):
            # ✅ keep the same temperature on retry
            retry_output = model_wrapper.prompt(prompts[idx], temperature=temperature)
            if retry_output and retry_output.strip() != "?" and len(retry_output.strip()) >= 3:
                df.loc[idx, "model_output"] = retry_output
            else:
                print(f"⚠️ Prompt {idx} failed even after retry. Prompt length: {len(prompts[idx])}")

    # Parse JSON and identify responses that need reprompting
    parsed = df["model_output"].apply(clean_and_parse_json)
    df["parsed_answer"] = parsed.apply(lambda x: x.get("answer"))
    df["parsed_confidence"] = pd.to_numeric(parsed.apply(lambda x: x.get("confidence")), errors="coerce")
    
    # Find indices that need reprompting for JSON formatting
    reprompt_indices = []
    for idx, result in enumerate(parsed):
        if result.get("needs_reprompt", False) and df.loc[idx, "model_output"] and df.loc[idx, "model_output"].strip():
            reprompt_indices.append(idx)
    
    if reprompt_indices:
        print(f"🔄 Found {len(reprompt_indices)} responses that need JSON reprompting...")
        for idx in tqdm(reprompt_indices, desc="Reprompting for JSON format"):
            original_output = df.loc[idx, "model_output"]
            reprompt = build_reprompt_for_json(original_output, dataset_name)
            
            # Try reprompting for proper JSON format (intentionally low/steady T)
            json_output = model_wrapper.prompt(reprompt, max_new_tokens=150, temperature=0.3)
            
            # Parse the reprompted output
            if json_output and json_output.strip():
                reparsed = clean_and_parse_json(json_output)
                if not reparsed.get("needs_reprompt", True):  # Successfully parsed
                    df.loc[idx, "model_output"] = json_output
                    df.loc[idx, "parsed_answer"] = reparsed.get("answer")
                    df.loc[idx, "parsed_confidence"] = reparsed.get("confidence")
                    continue
            
            # If reprompting failed, try to extract basic answer from original
            original_parsed = clean_and_parse_json(original_output)
            if original_parsed.get("answer") is not None:
                df.loc[idx, "parsed_answer"] = original_parsed.get("answer")
                if df.loc[idx, "parsed_confidence"] is None or pd.isna(df.loc[idx, "parsed_confidence"]):
                    df.loc[idx, "parsed_confidence"] = 50  # default fallback
                print(f"⚠️ Used fallback parsing for prompt {idx}")
            else:
                print(f"⚠️ Could not extract answer from prompt {idx} even after reprompting")

    # 🧾 add metadata columns for easy merging/analysis
    df["temperature"] = temperature
    df["dataset"] = dataset_name
    df["model_id"] = model_wrapper.model_id

    safe_model_id = model_wrapper.model_id.replace("/", "_").replace(" ", "_")
    Path(f"output/{folder_name}").mkdir(parents=True, exist_ok=True)
    output_file = f"output/{folder_name}{dataset_name}_verbalized_confidence_{safe_model_id}_T{temperature}{output_ending}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")




if __name__ == "__main__":

    """ 
    # Verbalized Confidence Elicitation Batch Script
    models = {
        "llama": load_llama,
        # "qwen": load_qwen,
        # "gemma": load_gemma,
        # "cogito": load_cogito,
        # "phi2": load_phi2
    }

    datasets = ["boolq", "squad", "trivia"] # ["squad", "trivia", "gsm8k", "boolq"]
    folder_name = "llm_confidence_elicitation/batch_1000_llama_3B/"
    Path(f"output/{folder_name}").mkdir(parents=True, exist_ok=True)

    for model_name, model_loader in models.items():
        print(f"\n🚀 Loading model: {model_name}")
        model = model_loader()

        for dataset in datasets:
            print(f"🔍 Running {model_name} on {dataset}...")
            run_verbalized_confidence_experiment(
                model_wrapper=model,
                n_samples=1000,
                dataset_name=dataset,
                folder_name=folder_name,
                output_ending=f"_{model_name}_batch_3B_1000",
                batch_size=4
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() """

    models = {
        # "llama": load_llama,
        "gemma": load_gemma,
        # other loaders...
    }

    datasets = ["squad", "trivia", "boolq"] # ["squad", "trivia", "gsm8k", "boolq"]
    folder_name = "llm_confidence_elicitation/batch_1000_llama_3B/gemma/"
    temperatures = [0.7]  # Sweep values

    for model_name, model_loader in models.items():
        print(f"\n🚀 Loading model: {model_name}")
        model = model_loader()

        for T in temperatures:
            for dataset in datasets:
                print(f"🔍 Running {model_name} on {dataset} at T={T} ...")
                run_verbalized_confidence_experiment(
                    model_wrapper=model,
                    n_samples=1000,
                    dataset_name=dataset,
                    folder_name=folder_name,
                    output_ending=f"_{model_name}_1000",
                    batch_size=4,
                    temperature=T
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

