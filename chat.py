from app.embeddings import load_embeddings
from app.vectorstore import load_vectorstore
from app.llm import load_model, generate_response
from app.prompt import build_prompt
from app.retriever import retrieve_documents


def main():
    print("=" * 50)
    print("DOST Scholarship AI Assistant")
    print("=" * 50)

    print("Loading embeddings...")
    embeddings = load_embeddings()

    print("Loading vector database...")
    vectorstore = load_vectorstore(embeddings)

    tokenizer, model = load_model()

    print("Assistant is ready!\n")

    while True:
        question = input("Ask a question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        # Retrieve relevant documents
        documents = retrieve_documents(vectorstore, question)

        # Build the prompt
        prompt = build_prompt(question, documents)

        # Generate the answer
        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt
        )

        # Display answer
        print("\nAssistant:")
        print(answer)

        # Display retrieved sources
        print("\nSources Used:")

        shown = set()

        for doc in documents:
            source = (
                doc.metadata.get("document", "Unknown Document"),
                doc.metadata.get("section", "General")
            )

            if source not in shown:
                print(f"• {source[0]} → {source[1]}")
                shown.add(source)

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()