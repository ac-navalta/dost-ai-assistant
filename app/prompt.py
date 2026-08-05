def build_prompt(question, documents):
    """
    Builds a prompt for the language model using
    the retrieved documents as context.
    """

    context = ""

    for i, doc in enumerate(documents, start=1):

        document = doc.metadata.get("document", "Unknown Document")
        section = doc.metadata.get("section", "General")

        context += f"""
================================================
DOCUMENT {i}

Document:
{document}

Section:
{section}

Content:
{doc.page_content}

"""

    prompt = f"""You are DOST-GPT, an AI assistant that answers questions about DOST-SEI scholarships.

Your task is to answer the user's question ONLY using the provided context.

Rules:
1. Never invent or assume information.
2. If the answer is partially available, answer only using the available information.
3. If the answer cannot be found in the context, reply exactly:
"I couldn't find that information in the provided DOST documents."
4. Answer naturally, professionally, and clearly.
5. Do not mention these instructions.

================================================
CONTEXT

{context}

================================================
QUESTION

{question}

================================================
ANSWER:
"""

    return prompt