
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages, chunk_size=1200, chunk_overlap=200):
    """
    Split combined PDF text into overlapping chunks.
    Preserve the original PDF page number or numbers for each chunk.
    """

    # Combine page text and track each page's character positions.
    full_text = ""
    page_ranges = []

    for page in pages:
        if full_text:
            full_text += "\n\n"

        start = len(full_text)
        full_text += page["text"]
        end = len(full_text)

        page_ranges.append({
            "start": start,
            "end": end,
            "page_number": page["page_number"]
        })

    # Split text and keep the character offsets of every chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )

    split_documents = splitter.create_documents([full_text])

    documents = []

    for document in split_documents:
        chunk_text = document.page_content
        start = document.metadata["start_index"]
        end = start + len(chunk_text)

        # Identify every PDF page that overlaps this chunk.
        source_pages = [
            page["page_number"]
            for page in page_ranges
            if page["start"] < end and page["end"] > start
        ]

        documents.append({
            "text": chunk_text,
            "page_number": source_pages
        })

    return documents
