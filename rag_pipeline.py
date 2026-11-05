
from src.retriever import Retriever
from src.generator import generate_answer
from src.hallucination import HallucinationChecker


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.hallucination_checker = HallucinationChecker()

    def ask(self, question):
        # 1. Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            question,
            k=5
        )

        # 2. Generate an answer using the retrieved context
        answer = generate_answer(
            question,
            retrieved_chunks
        )

        # 3. Check the answer against the retrieved context
        verification = self.hallucination_checker.check(
            answer,
            retrieved_chunks
        )

        # 4. Return answer, sources, and verification results
        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "hallucination_check": verification
        }
