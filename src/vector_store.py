
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """
    Store text chunks and their source page numbers in FAISS.
    """

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {"page_number": chunk["page_number"]}
        for chunk in chunks
    ]

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    return vector_store
