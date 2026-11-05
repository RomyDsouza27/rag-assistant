
from pathlib import Path

from src.ingestion import load_pdfs, create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.pdf_visual_extractor import extract_pdf_visual_content


DOCUMENTS_FOLDER = "data/documents"
VISUAL_OUTPUT_FOLDER = "data/processed_pages"

# Keep None for OCR and caption extraction only.
# To enable diagram interpretation, replace None with the
# exact name of a vision-capable model installed in Ollama.
VISION_MODEL = "qwen2.5vl:3b"

# Keep visual chunks short enough for effective embedding.
VISUAL_CHUNK_SIZE = 1000
VISUAL_CHUNK_OVERLAP = 150


def split_visual_record(record):
    """Split visual-page text while retaining its metadata."""
    text = record.get("text", "").strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + VISUAL_CHUNK_SIZE, len(text))

        # Prefer ending at a paragraph or word boundary.
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary < start + VISUAL_CHUNK_SIZE // 2:
                boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        piece = text[start:end].strip()

        if piece:
            chunks.append({
                **record,
                "text": piece,
                "content_type": record.get(
                    "content_type", "visual_page"
                ),
            })

        if end >= len(text):
            break

        start = max(end - VISUAL_CHUNK_OVERLAP, start + 1)

    return chunks


print("Loading all research papers...")
documents = load_pdfs(DOCUMENTS_FOLDER)
print(f"Loaded {len(documents)} pages in total")

print("\nCreating regular text chunks...")
text_chunks = create_chunks(documents)
print(f"Created {len(text_chunks)} regular text chunks")

if not text_chunks:
    raise ValueError(
        "No text chunks were created. Check the PDF files."
    )

print("\nExtracting captions, OCR text, and visual descriptions...")
visual_records = []
pdf_paths = sorted(Path(DOCUMENTS_FOLDER).glob("*.pdf"))

for index, pdf_path in enumerate(pdf_paths, start=1):
    print(
        f"[{index}/{len(pdf_paths)}] Processing "
        f"{pdf_path.name}"
    )

    try:
        records = extract_pdf_visual_content(
            pdf_path=str(pdf_path),
            output_dir=VISUAL_OUTPUT_FOLDER,
            vision_model=VISION_MODEL,
        )
        visual_records.extend(records)

    except Exception as exc:
        print(
            f"Visual extraction failed for "
            f"{pdf_path.name}: {exc}"
        )

visual_chunks = []

for record in visual_records:
    visual_chunks.extend(split_visual_record(record))

print(f"Extracted {len(visual_records)} visual-page records")
print(f"Created {len(visual_chunks)} visual chunks")

# Combine both types of searchable content.
all_chunks = text_chunks + visual_chunks

print(f"\nTotal chunks before validation: {len(all_chunks)}")

valid_chunks = []
texts = []

print("\nValidating text chunks...")

for i, chunk in enumerate(all_chunks):
    text = chunk.get("text")

    if not isinstance(text, str) or not text.strip():
        print(f"Skipping invalid chunk {i}")
        continue

    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        print(f"Cleaning malformed Unicode in chunk {i}")
        text = text.encode(
            "utf-8", errors="replace"
        ).decode("utf-8")

    if not text.strip():
        print(f"Skipping empty chunk {i}")
        continue

    cleaned_chunk = {**chunk, "text": text}
    valid_chunks.append(cleaned_chunk)
    texts.append(text)

print("Valid chunks:", len(texts))
print("Skipped chunks:", len(all_chunks) - len(valid_chunks))

if not texts:
    raise ValueError("No valid text chunks remain.")

print("\nLoading embedding model...")
embedding_model = EmbeddingModel()

print("\nGenerating embeddings...")
embeddings = embedding_model.encode(texts)

print("Embedding shape:", embeddings.shape)

if len(embeddings) != len(valid_chunks):
    raise ValueError(
        "Embedding count does not match chunk count."
    )

print("\nBuilding FAISS index...")
vector_store = VectorStore()
vector_store.create(embeddings, valid_chunks)

print("\nFAISS index created successfully!")
print(f"Indexed {len(valid_chunks)} chunks.")

visual_count = sum(
    1 for chunk in valid_chunks
    if chunk.get("content_type") == "visual_page"
)

print(f"Visual chunks indexed: {visual_count}")
print(f"Regular text chunks indexed: {len(valid_chunks) - visual_count}")
