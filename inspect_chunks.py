
from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import chunk_pages

pages = extract_text_from_pdf("sample.pdf")
chunks = chunk_pages(pages)

for i, chunk in enumerate(chunks):
    text = chunk["text"].lower()

    if "5 kingdom" in text or "whittaker" in text:
        print(f"\n--- Chunk {i + 1} ---")
        print("Source page:", chunk["page_number"])
        print(chunk["text"])

        # Also inspect the next chunk, if one exists
        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            print(f"\n--- Next Chunk {i + 2} ---")
            print("Source page:", next_chunk["page_number"])
            print(next_chunk["text"])
