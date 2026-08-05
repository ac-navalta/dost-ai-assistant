from app.embeddings import load_embeddings
from app.vectorstore import load_vectorstore
from app.retriever import retrieve_documents

embeddings = load_embeddings()

vectorstore = load_vectorstore(embeddings)

query = input("Ask a question: ")

results = retrieve_documents(vectorstore, query, k=1)

print("\nRetrieved Chunks\n")
print("=" * 60)

for i, doc in enumerate(results, start=1):
    print(f"\nChunk {i}")
    print("-" * 40)
    print(doc.page_content)
    print("\nMetadata:", doc.metadata)