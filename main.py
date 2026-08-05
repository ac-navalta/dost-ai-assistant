from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

texts = [
    "Hello",
    "Hello world",
    "Artificial Intelligence is fascinating."
]

inputs = tokenizer(
    texts,
    padding=True,
    return_tensors="pt"
)

print(inputs["input_ids"])
print()
print(inputs["attention_mask"])
