import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:
    @staticmethod
    def cosine_similarity_matrix(
        X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        return cosine_similarity(X, Y)

    @staticmethod
    def precision_at_k(
        relevant: list[int], retrieved: list[int], k: int = 5
    ) -> float:
        if not retrieved:
            return 0.0
        retrieved_k = retrieved[:k]
        hits = len(set(relevant) & set(retrieved_k))
        return hits / min(k, len(retrieved_k))

    @staticmethod
    def mean_average_precision(
        query_relevant: list[list[int]],
        query_retrieved: list[list[int]],
    ) -> float:
        aps = []
        for rel, ret in zip(query_relevant, query_retrieved):
            hits = 0
            sum_prec = 0.0
            for k, doc in enumerate(ret[:len(rel)], start=1):
                if doc in rel:
                    hits += 1
                    sum_prec += hits / k
            aps.append(sum_prec / len(rel) if rel else 0.0)
        return np.mean(aps)
