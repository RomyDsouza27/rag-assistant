
from src.ingestion import load_pdf

pdf_path = "data/documents/research_paper.pdf"
pages = load_pdf(pdf_path)

keywords = [
    "objective",
    "aim of this paper",
    "purpose",
    "proposed system",
    "in this paper",
    "language detection",
]

for page in pages:
    text = page["text"]
    if any(word in text.lower() for word in keywords):
        print(f"\n{'=' * 60}")
        print(f"PAGE {page['page']}")
        print(text[:2500])
