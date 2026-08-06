from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE


def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    print("Loading language model...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )

    print("Model loaded successfully!\n")

    return tokenizer, model


def generate_response(model, tokenizer, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer