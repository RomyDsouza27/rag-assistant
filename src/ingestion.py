
import re
from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):
    """Extract text from one PDF and preserve page metadata."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text,
                "source": pdf_path.name,
                "page": page_number,
            })

    if not documents:
        raise ValueError(
            f"No text extracted from {pdf_path.name}. "
            "The PDF may be scanned or image-based."
        )

    print(f"{pdf_path.name}: extracted {len(documents)} pages.")
    return documents


def load_pdfs(folder_path="data/documents"):
    """Extract text from every PDF in a folder."""
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {folder_path}"
        )

    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {folder_path}"
        )

    all_documents = []
    skipped_files = []

    for pdf_path in pdf_files:
        try:
            documents = load_pdf(pdf_path)
            all_documents.extend(documents)
        except Exception as error:
            skipped_files.append(pdf_path.name)
            print(f"Warning: Skipping {pdf_path.name}: {error}")

    if not all_documents:
        raise ValueError(
            "No readable PDF text was extracted. "
            "Check the PDFs and their text extraction."
        )

    print(f"\nTotal PDFs found: {len(pdf_files)}")
    print(f"Total pages extracted: {len(all_documents)}")
    print(f"PDFs skipped: {len(skipped_files)}")

    return all_documents


def chunk_text(text, chunk_size=1000, overlap=150):
    """Create overlapping text chunks while preserving paragraphs."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Use chunk_size > 0 and 0 <= overlap < chunk_size."
        )

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    step = chunk_size - overlap

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            start = 0
            while start < len(paragraph):
                piece = paragraph[start:start + chunk_size].strip()

                if piece:
                    chunks.append(piece)

                start += step

            continue

        candidate = f"{current} {paragraph}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = f"{current[-overlap:] if current else ''} {paragraph}".strip()

    if current:
        chunks.append(current)

    return chunks


def create_chunks(documents):
    """Create searchable chunks and preserve paper/page metadata."""
    chunks = []

    for document in documents:
        text = document["text"]

        # Remove common repeated journal headers when present.
        text = re.sub(
            r"(?im)^.*(?:INTERNATIONAL JOURNAL OF INNOVATIVE "
            r"RESEARCH IN TECHNOLOGY|IJIRT\s*\d+|"
            r"ISSN:\s*\d[\d-]*|©\s*May\s+2024).*$",
            "",
            text,
        )

        for chunk in chunk_text(text):
            letters = re.sub(r"[^A-Za-z]", "", chunk)

            if len(letters) < 100:
                continue

            chunks.append({
                "text": chunk,
                "source": document["source"],
                "page": document["page"],
            })

    print(f"Created {len(chunks)} text chunks.")
    return chunks
