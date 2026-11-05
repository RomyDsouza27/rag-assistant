
import re
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.vector_store.load()

    @staticmethod
    def normalize_text(text):
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            text.lower()
        ).strip()

    def find_source_in_question(self, question):
        """Find a PDF filename explicitly mentioned in the question."""
        question_normalized = self.normalize_text(question)

        sources = {
            chunk.get("source")
            for chunk in self.vector_store.metadata
            if chunk.get("source")
        }

        for source in sorted(sources, key=len, reverse=True):
            stem = Path(source).stem
            normalized_stem = self.normalize_text(stem)

            if (
                normalized_stem
                and normalized_stem in question_normalized
            ):
                return source

        return None

    @staticmethod
    def is_objective_question(question):
        return bool(
            re.search(
                r"\b(objective|objectives|aim|aims|purpose|"
                r"goal|goals|main idea)\b",
                question.lower()
            )
        )

    @staticmethod
    def rerank_score(item, objective_question=False):
        text = item.get("text", "").lower()
        score = float(item.get("score", 0.0))

        if objective_question:
            # Prefer passages likely to describe research objectives.
            if re.search(r"\babstract\b", text):
                score += 0.10

            objective_patterns = [
                r"\baim(?:s|ed)? to\b",
                r"\bobjective(?:s)? of (?:this|the) (?:study|paper|research)\b",
                r"\bpurpose of this (?:study|paper|research)\b",
                r"\bthis paper (?:proposes|presents|investigates|examines)\b",
                r"\bthis study (?:aims|investigates|examines|seeks)\b",
                r"\bwe propose\b",
                r"\bwe investigate\b",
            ]

            if any(
                re.search(pattern, text)
                for pattern in objective_patterns
            ):
                score += 0.20

        return score

    def retrieve(self, question, k=5):
        metadata = self.vector_store.metadata

        if not metadata:
            return []

        source = self.find_source_in_question(question)

        # Remove the filename from the semantic query when possible.
        semantic_question = question

        if source:
            semantic_question = re.sub(
                re.escape(source),
                " ",
                question,
                flags=re.IGNORECASE
            )

        query_embedding = self.embedding_model.encode(
            [semantic_question]
        )[0]

        if source:
            # Search the full index so relevant chunks from the
            # requested paper are not lost behind other papers.
            results = self.vector_store.search(
                query_embedding,
                k=len(metadata)
            )

            results = [
                item for item in results
                if item.get("source") == source
            ]
        else:
            candidate_count = min(30, len(metadata))

            results = self.vector_store.search(
                query_embedding,
                k=candidate_count
            )

        objective_question = self.is_objective_question(question)

        reranked = []

        for item in results:
            updated_item = item.copy()
            updated_item["score"] = self.rerank_score(
                updated_item,
                objective_question
            )
            reranked.append(updated_item)

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return reranked[:k]
