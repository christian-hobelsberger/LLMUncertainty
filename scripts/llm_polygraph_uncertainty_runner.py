import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

from lm_polygraph import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import CocoaMSP


# --------------- MODEL WRAPPER ----------------

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


# ----------- MODEL LOADERS -------------

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


# ----------- DATASET HANDLING -------------

def load_dataset(dataset_name: str, sample_size: int) -> pd.DataFrame:
    dataset_paths = {
        "squad": "data/SQuAD2_merged.csv",
        "trivia": "data/trivia_all.csv",
        "boolq": "data/boolQ.csv",
        "gsm8k": "data/gsm8k_cleaned.csv"
    }

    if dataset_name not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    df = pd.read_csv(dataset_paths[dataset_name])
    return df.sample(n=sample_size, random_state=42).reset_index(drop=True)


def build_input_text(row, dataset_name: str) -> str:
    if dataset_name == "squad":
        return f"Context: {row['context']}\nQuestion: {row['question']}"
    elif dataset_name == "boolq":
        return f"You are a reading comprehension assistant. Given a passage and a yes/no question, respond with a single word: either 'yes' or 'no' — but not both.\nPassage: {row['passage']}\nQuestion: {row['question']}"
    elif dataset_name == "trivia":
        return f"Question: {row['question']}"
    elif dataset_name == "gsm8k":
        return f"Solve: {row['question']}"
    else:
        raise ValueError("Unsupported dataset for prompt building.")


def is_correct(pred, true):
    if pd.isna(true) or pd.isna(pred):
        return False
    return str(pred).strip().lower() == str(true).strip().lower()


# ----------- MAIN EXPERIMENT -------------

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
    polygraph_model = WhiteboxModel(raw_model, tokenizer)

    ue_method = ue_method_class()

    print(f"\n📊 Loading {dataset_name} dataset with {sample_size} samples...")
    df = load_dataset(dataset_name, sample_size)
    df["input_text"] = df.apply(lambda row: build_input_text(row, dataset_name), axis=1)

    print(f"🧠 [{model_name_safe}] Estimating uncertainty with {ue_method_class.__name__}...")

    model_answers = []
    uncertainties = []

    for text in tqdm(df["input_text"].tolist(), desc=f"{model_name_safe} on {dataset_name}"):
        output = estimate_uncertainty(polygraph_model, ue_method, text)

        model_answers.append(output.generation_text.strip() if output.generation_text else "")
        uncertainties.append(output.uncertainty)

    df["model_output"] = output
    df["model_answer"] = model_answers
    df["uncertainty_score"] = uncertainties

    # Optional: Include ground truth and match
    if dataset_name == "squad":
        df["ground_truth"] = df["answer_text"]
    elif dataset_name in ["trivia", "gsm8k"]:
        df["ground_truth"] = df["answer"]
    elif dataset_name == "boolq":
        df["ground_truth"] = df["answer"].map(lambda x: str(x).strip().lower())

    if "ground_truth" in df.columns:
        df["match"] = df.apply(lambda row: is_correct(row["model_answer"], row["ground_truth"]), axis=1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{dataset_name}_{model_name_safe}_{ue_method_class.__name__}_{sample_size}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved results to: {output_file}")


# ---------------- MAIN ----------------

def main():
    # Define models to iterate over
    models = {
        "llama": load_llama,
        # "qwen": load_qwen,
        # "phi2": load_phi2,
        # "gemma": load_gemma,
        # "cogito": load_cogito,  # Optional if GPU memory allows
    }

    datasets = ["boolq"] # ["squad", "trivia", "boolq", "gsm8k"]
    sample_size = 500
    ue_method_class = CocoaMSP
    output_dir = "output/polygraph/CocoaMSP/"

    for model_name, model_loader in models.items():
        print(f"\n🚀 Running model: {model_name}")
        model_wrapper = model_loader()

        for dataset in datasets:
            run_polygraph_inference(
                model_wrapper=model_wrapper,
                ue_method_class=ue_method_class,
                dataset_name=dataset,
                sample_size=sample_size,
                output_dir=output_dir
            )

        del model_wrapper
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
