import re
from langchain_core.documents import Document


def split_documents(documents):

    chunks = []

    for doc in documents:

        text = doc.page_content

        # Split every time we encounter a level-2 heading (##)
        sections = re.split(r'(?=^##\s)', text, flags=re.MULTILINE)

        title = ""

        # Capture the H1 title
        match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if match:
            title = match.group(1)

        for section in sections:

            section = section.strip()

            if not section:
                continue

            metadata = dict(doc.metadata)
            metadata["document"] = title

            # Find the section name
            section_match = re.search(r'^##\s+(.+)$', section, re.MULTILINE)

            if section_match:
                metadata["section"] = section_match.group(1)

            chunks.append(
                Document(
                    page_content=section,
                    metadata=metadata
                )
            )

    return chunks