from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import Counter
from typing import List, Dict


class LLMWrapper:
    def __init__(self, model_id: str, trust_remote_code: bool = False):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=trust_remote_code)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def prompt(self, text: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        output = self.pipe(text, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        return output[0]['generated_text'].strip()

    def batch_prompt(self, prompts: List[str], max_new_tokens: int = 100, temperature: float = 0.7) -> List[str]:
        return [self.prompt(prompt, max_new_tokens, temperature) for prompt in prompts]

    def sample_variants(self, prompt: str, n: int = 5, max_new_tokens: int = 100, temperature: float = 0.7) -> Dict[str, int]:
        responses = [self.prompt(prompt, max_new_tokens, temperature) for _ in range(n)]
        return Counter(responses)


# Predefined model wrappers for convenience

def load_llama():
    return LLMWrapper("meta-llama/Llama-3.2-1B-Instruct")

def load_qwen():
    return LLMWrapper("Qwen/Qwen3-8B", trust_remote_code=True)

def load_gemma():
    return LLMWrapper("google/gemma-3-1b-it")


if __name__ == "__main__":
    llama = load_llama()
    print("LLaMA sample:", llama.prompt("Q: What is the capital of France? Include confidence.\nA:"))

    qwen = load_qwen()
    print("Qwen sample:", qwen.prompt("User: Who painted the Mona Lisa? Include your confidence.\nAssistant:"))

    gemma = load_gemma()
    print("Gemma sample:", gemma.prompt("Question: What is the boiling point of water in Celsius? Include certainty.\nAnswer:"))
