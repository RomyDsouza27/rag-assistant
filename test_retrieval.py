from src.retriever import Retriever


retriever = Retriever()

question = input("Enter your question: ")

results = retriever.retrieve(
    question,
    k=5
)

for i, result in enumerate(results, start=1):

    print("\n" + "=" * 60)

    print(f"Result {i}")

    print("Score:", result["score"])

    print("Source:", result["source"])

    print("Page:", result["page"])

    print("\nText:")

    print(result["text"])