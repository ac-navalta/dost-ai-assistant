from app.loader import load_documents
from app.splitter import split_documents
from app.embeddings import load_embeddings
from app.vectorstore import (
    create_vectorstore,
    save_vectorstore
)

documents = load_documents("data/knowledge_base")

chunks = split_documents(documents)

embeddings = load_embeddings()

vectorstore = create_vectorstore(
    chunks,
    embeddings
)

save_vectorstore(vectorstore)

print("Vector database created successfully!")