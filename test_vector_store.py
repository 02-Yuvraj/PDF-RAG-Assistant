
from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import chunk_pages
from src.embedding_generator import create_embedding_model
from src.vector_store import create_vector_store
from src.answer_generator import generate_answer
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Step 1: Extract text from the PDF
pages = extract_text_from_pdf("sample.pdf")

# Step 2: Split the text into chunks
chunks = chunk_pages(pages)

# Step 3: Load the embedding model
embedding_model = create_embedding_model()

# Step 4: Create the FAISS vector store
vector_store = create_vector_store(chunks, embedding_model)

# Step 5: Check the number of stored vectors
print("Total PDF pages:", len(pages))
print("Total text chunks:", len(chunks))
print("Total vectors in FAISS:", vector_store.index.ntotal)


# Step 6: Prepare documents for keyword search
documents = [
    Document(
        page_content=chunk["text"],
        metadata={"page_number": chunk["page_number"]}
    )
    for chunk in chunks
]

# Step 7: Create keyword retriever
keyword_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)

# Step 8: Create semantic retriever
semantic_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Step 9: Combine both retrievers
hybrid_retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, semantic_retriever],
    weights=[0.5, 0.5]
)

# Step 10: Search and return 3 results
question = "What are the five kingdom classifications proposed by R. H. Whittaker?"
results = hybrid_retriever.invoke(question)[:3]

print("\nQuestion:", question)

for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Source page(s):", document.metadata["page_number"])
    print("Text:", document.page_content)

# Step 11: Generate an answer using Groq
answer = generate_answer(question, results)

print("\n" + "=" * 50)
print("GENERATED ANSWER")
print("=" * 50)
print(answer)