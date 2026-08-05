from langchain_community.document_loaders import TextLoader
from pathlib import Path

def load_documents(folder):

    documents = []

    for file in Path(folder).glob("*.md"):

        loader = TextLoader(
            str(file),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    return documents