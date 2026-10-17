
from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import chunk_pages

# Step 1: Extract text from the PDF
pages = extract_text_from_pdf("sample.pdf")

# Step 2: Split the extracted text into chunks
chunks = chunk_pages(pages)

# Step 3: Print the summary
print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

# Step 4: Display the first five chunks
for i, chunk in enumerate(chunks[:5], start=1):
    print(f"\n--- Chunk {i} ---")
    print("Source page:", chunk["page_number"])
    print("Text:", chunk["text"][:200])
