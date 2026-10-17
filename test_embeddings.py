
from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import chunk_pages
from src.embedding_generator import (
    create_embedding_model,
    generate_embeddings
)

# Step 1: Extract PDF text
pages = extract_text_from_pdf("sample.pdf")

# Step 2: Create chunks
chunks = chunk_pages(pages)

# Step 3: Load the embedding model
model = create_embedding_model()

# Step 4: Generate embeddings
embeddings = generate_embeddings(chunks, model)

# Step 5: Display the results
print("Total pages:", len(pages))
print("Total chunks:", len(chunks))
print("Total embeddings:", len(embeddings))

if embeddings:
    print("Dimensions per embedding:", len(embeddings[0]))
    print("First embedding preview:", embeddings[0][:10])
