
import json
from pathlib import Path

metadata_path = Path("vectorstore/metadata.json")

if not metadata_path.exists():
    print("Metadata file not found. Run python build_index.py first.")
    raise SystemExit(1)

with metadata_path.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Total chunks:", len(metadata))

for i, chunk in enumerate(metadata):
    text = chunk.get("text", "")
    if (
        chunk.get("page") == 1
        or "robust system for" in text.lower()
        or "automated language detection" in text.lower()
    ):
        print(f"\nChunk {i}")
        print("Page:", chunk.get("page"))
        print("Text:", text[:1000])
