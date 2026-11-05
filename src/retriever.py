
import re
from collections import defaultdict
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        candidate_k=40,
        default_k=5,
        max_chunks_per_source=2,
    ):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.vector_store.load()

        self.candidate_k = candidate_k
        self.default_k = default_k
        self.max_chunks_per_source = max_chunks_per_source

    @staticmethod
    def normalize(text):
        """Normalize text for filename and keyword matching."""
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def find_requested_source(self, question):
        """
        Return a matching PDF filename only when the user explicitly
        mentions a filename in the question.
        """
        normalized_question = self.normalize(question)

        sources = {
            str(chunk.get("source", ""))
            for chunk in self.vector_store.metadata
            if chunk.get("source")
        }

        for source in sources:
            filename = Path(source).name
            stem = Path(filename).stem

            normalized_filename = self.normalize(filename)
            normalized_stem = self.normalize(stem)

            if (
                normalized_filename in normalized_question
                or (
                    len(normalized_stem) >= 5
                    and normalized_stem in normalized_question
                )
            ):
                return source

        return None

    @staticmethod
    def lexical_overlap(question, text):
        """
        Calculate simple query-term overlap.
        This is a reranking feature, not a factuality score.
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "of", "in", "on", "at", "to", "for", "and",
            "or", "what", "which", "who", "how", "why",
            "does", "do", "did", "this", "that", "with",
            "from", "about", "paper", "research", "please",
            "summarize", "explain", "discuss", "describe",
        }

        query_terms = {
            word for word in Retriever.normalize(question).split()
            if len(word) > 2 and word not in stop_words
        }

        text_terms = set(Retriever.normalize(text).split())

        if not query_terms:
            return 0.0

        return len(query_terms & text_terms) / len(query_terms)

    def retrieve(self, question, k=None):
        """
        Retrieve relevant chunks across the complete collection,
        unless the question explicitly names a source PDF.
        """
        if k is None:
            k = self.default_k

        if not question or not question.strip():
            return []

        if not self.vector_store.metadata:
            return []

        # Encode the question using the same model as the document index.
        query_embedding = self.embedding_model.encode(
            [question.strip()]
        )[0]

        # Search broadly before reranking.
        candidate_count = min(
            self.candidate_k,
            len(self.vector_store.metadata)
        )

        candidates = self.vector_store.search(
            query_embedding,
            k=candidate_count
        )

        if not candidates:
            return []

        # Filter by source only when the user explicitly names a PDF.
        requested_source = self.find_requested_source(question)

        if requested_source:
            matching = [
                chunk for chunk in candidates
                if chunk.get("source") == requested_source
            ]

            # If that PDF was not in the initial candidates, search the
            # index more broadly and filter again.
            if not matching:
                all_candidates = self.vector_store.search(
                    query_embedding,
                    k=len(self.vector_store.metadata)
                )
                matching = [
                    chunk for chunk in all_candidates
                    if chunk.get("source") == requested_source
                ]

            candidates = matching

        if not candidates:
            return []

        # Combine semantic similarity with keyword overlap.
        for chunk in candidates:
            similarity = float(chunk.get("score", 0.0))
            overlap = self.lexical_overlap(
                question, chunk.get("text", "")
            )

            chunk["_retrieval_score"] = (
                0.80 * similarity + 0.20 * overlap
            )

        candidates.sort(
            key=lambda chunk: chunk["_retrieval_score"],
            reverse=True
        )

        # Keep results relevant while allowing multiple papers to appear.
        results = []
        source_counts = defaultdict(int)

        for chunk in candidates:
            source = chunk.get("source", "Unknown source")

            if source_counts[source] >= self.max_chunks_per_source:
                continue

            # Preserve the original similarity score for the UI.
            result = {
                key: value
                for key, value in chunk.items()
                if key != "_retrieval_score"
            }

            result["retrieval_score"] = chunk["_retrieval_score"]
            results.append(result)
            source_counts[source] += 1

            if len(results) >= k:
                break

        return results
