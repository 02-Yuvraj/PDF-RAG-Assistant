
from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


def generate_embeddings(chunks, model):
    """
    Convert each text chunk into a numerical vector.
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.embed_documents(texts)

    return embeddings
