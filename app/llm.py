from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

    print("Model loaded successfully!")

    return tokenizer, model