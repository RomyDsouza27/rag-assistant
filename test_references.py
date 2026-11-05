
from src.retriever import Retriever

retriever = Retriever()

question = "How many references are mentioned in the paper?"

print("Testing reference question detection:")
print(retriever.is_reference_count_question(question))

print("\nSearching for the References section...")
chunks = retriever.get_reference_chunks()

print("Reference chunks found:", len(chunks))

for chunk in chunks:
    print("\nPage:", chunk.get("page", "Unknown"))
    print(chunk.get("text", "")[:500])
