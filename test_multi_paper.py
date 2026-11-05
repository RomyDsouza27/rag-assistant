
from src.retriever import Retriever

retriever = Retriever()

question = "What are the main objectives of the research?"
results = retriever.retrieve(question, k=5)

print("\nQuestion:", question)
print("Retrieved chunks:", len(results))

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Source:", result.get("source", "Unknown"))
    print("Page:", result.get("page", "Unknown"))
    print("Score:", result.get("score", "Unknown"))
    print("Text:", result.get("text", "")[:500])
