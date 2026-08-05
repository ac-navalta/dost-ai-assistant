import torch
import transformers
import sentence_transformers
import langchain
import faiss

print("=" * 40)
print("Environment Check")
print("=" * 40)

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Sentence Transformers: {sentence_transformers.__version__}")
print(f"LangChain: {langchain.__version__}")
print(f"FAISS: {faiss.__version__}")

print()

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")