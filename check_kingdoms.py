
from src.pdf_processor import extract_text_from_pdf

pages = extract_text_from_pdf("sample.pdf")

keywords = [
    "5 kingdom classifications",
    "monera",
    "protista",
    "fungi",
    "plantae",
    "animalia",
    "whittaker"
]

for page in pages:
    text = page["text"]
    text_lower = text.lower()

    if any(keyword in text_lower for keyword in keywords):
        print(f"\n{'=' * 60}")
        print(f"PDF PAGE: {page['page_number']}")
        print("=" * 60)
        print(text)
