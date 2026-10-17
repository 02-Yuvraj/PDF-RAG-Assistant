
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


def generate_answer(question, retrieved_documents):
    """
    Generate an answer using only the retrieved PDF passages.
    Include source page numbers and acknowledge insufficient evidence.
    """

    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY not found. Check your .env file."
        )

    if not retrieved_documents:
        return "I couldn't find relevant information in the PDF to answer this question."

    # Prepare the retrieved PDF passages with their source pages.
    context_parts = []

    for document in retrieved_documents:
        pages = document.metadata.get("page_number", [])
        if isinstance(pages, int):
            pages = [pages]

        page_labels = ", ".join(str(page) for page in pages)

        context_parts.append(
            f"[PDF page(s): {page_labels}]\n"
            f"{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Initialize the same model used in the successful connection test.
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    messages = [
        (
            "system",
            """
You are a PDF question-answering assistant.

Rules:
1. Answer using ONLY the supplied PDF passages.
2. Do not use outside knowledge or invent missing information.
3. Cite supporting PDF page numbers using [Page N].
4. Only cite page numbers explicitly supplied with the passages.
5. If the passages do not contain enough evidence, clearly say:
   "The retrieved PDF passages do not contain enough information
   to answer this question completely."
6. Distinguish between information explicitly stated in the PDF
   and information that cannot be confirmed from the passages.
7. Be clear and concise.
"""
        ),
        (
            "human",
            f"""
Question:
{question}

Retrieved PDF passages:
{context}

Answer the question using only these passages.
"""
        )
    ]

    response = llm.invoke(messages)

    return response.content
