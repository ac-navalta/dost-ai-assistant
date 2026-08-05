from app.config import TOP_K

def retrieve_documents(vectorstore, query, k=TOP_K):
    return vectorstore.similarity_search(query, k=k)