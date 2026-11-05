import os
import json
import faiss
import numpy as np


class VectorStore:

    def __init__(self, index_path="vectorstore/index.faiss",
                 metadata_path="vectorstore/metadata.json"):

        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata = []

    def create(self, embeddings, chunks):

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.metadata = chunks

        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self):

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

    def search(self, query_embedding, k=5):

        query_embedding = np.array(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            result = self.metadata[index].copy()

            result["score"] = float(score)

            results.append(result)

        return results