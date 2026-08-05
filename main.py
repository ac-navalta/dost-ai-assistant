from app.llm import load_model

print("=" * 40)
print("DOST Scholarship AI Assistant")
print("=" * 40)

tokenizer, model = load_model()

print("\nModel is ready!\n")

prompt = "Hello! Introduce yourself in one sentence."

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=80
)

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

print("\nAssistant:")
print(response)