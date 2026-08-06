from app.embeddings import load_embeddings
from app.vectorstore import load_vectorstore
from app.llm import load_model, generate_response
from app.memory import ConversationMemory
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

    memory = ConversationMemory(max_turns=3)

    print("Assistant is ready!\n")

    while True:

        question = input("Ask a question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            memory.clear()
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        # Retrieve relevant documents
        documents = retrieve_documents(vectorstore, question)

        # Build system prompt
        system_prompt = build_prompt(documents)

        # Build chat messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        messages.extend(memory.get_messages())

        messages.append({
            "role": "user",
            "content": question
        })

        # Convert to Qwen chat format
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate answer
        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=chat_prompt
        )

        # Store to memory
        memory.add("user", question)
        memory.add("assistant", answer)

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