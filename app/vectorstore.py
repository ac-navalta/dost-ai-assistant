from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embeddings):

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore

def save_vectorstore(vectorstore):

    vectorstore.save_local(
        "data/vectorstore"
    )

def load_vectorstore(embeddings):
    vectorstore = FAISS.load_local(
        "data/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore