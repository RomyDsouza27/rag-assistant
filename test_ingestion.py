from src.ingestion import load_pdf, create_chunks


pdf_path = "data/documents/research_paper.pdf"

documents = load_pdf(pdf_path)

print("Pages loaded:", len(documents))

chunks = create_chunks(documents)

print("Chunks created:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0]["text"])

print("\nSource:")
print(chunks[0]["source"])

print("\nPage:")
print(chunks[0]["page"])