
import ollama


def generate_answer(question, retrieved_chunks):
    # Handle cases where retrieval returns no results
    if not retrieved_chunks:
        return (
            "I could not find relevant information in the research "
            "documents to answer this question. Please try rephrasing "
            "your question."
        )

    # Build context from the retrieved research-paper chunks
    context_parts = []

    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown source")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    if not context:
        return (
            "The retrieved documents do not contain readable text "
            "that can support an answer."
        )

    prompt = f"""
You are an intelligent research assistant that answers questions
using a collection of research papers.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

INSTRUCTIONS:
1. Answer the user's question directly using the relevant evidence.
2. The user does not need to mention a paper filename.
3. Search the provided evidence for information relevant to the
   question, regardless of which paper it comes from.
4. For requests to summarize a paper, explain its objective,
   methodology, and findings when those details are available.
5. For broad questions, synthesize relevant evidence from multiple
   papers when the retrieved context supports doing so.
6. Distinguish findings from different papers instead of combining
   them into one unsupported claim.
7. Cite the source filename and page number for important claims.
8. Never invent results, statistics, authors, methodologies, or facts.
9. If a requested detail is absent from the evidence, say that it
   is not available in the retrieved passages.
10. If the evidence is insufficient to answer the question, clearly
    explain this limitation rather than guessing.
11. Do not claim to have read a complete paper when only excerpts
    are provided.

Write a clear, organized answer appropriate for a research assistant.
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"].strip()
