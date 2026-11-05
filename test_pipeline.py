
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

question = input("Enter your question: ")

result = pipeline.ask(question)

print("\n" + "=" * 50)
print("GENERATED ANSWER")
print("=" * 50)
print(result["answer"])

print("\n" + "=" * 50)
print("HALLUCINATION CHECK")
print("=" * 50)

check = result["hallucination_check"]

print("Supported:", check["supported"])
print("Score:", check["score"])
print("Warning:", check["warning"])

if check["unsupported_sentences"]:
    print("\nPotentially unsupported claims:")
    for sentence in check["unsupported_sentences"]:
        print("-", sentence)

print("\n" + "=" * 50)
print("SOURCES")
print("=" * 50)

for source in result["sources"]:
    print(
        f"Page: {source.get('page', 'Unknown')} | "
        f"File: {source.get('source', 'Unknown')}"
    )
