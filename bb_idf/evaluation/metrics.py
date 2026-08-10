import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:
    @staticmethod
    def cosine_similarity_matrix(
        X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        return cosine_similarity(X, Y)

    @staticmethod
    def extract_keywords(
        weight_matrix: np.ndarray,
        vocabulary: dict[str, int],
        top_n: int = 5,
    ) -> list[list[tuple[str, float]]]:
        idx_to_term = {v: k for k, v in vocabulary.items()}
        results = []
        for row in weight_matrix:
            top_idx = row.argsort()[::-1][:top_n]
            results.append([
                (idx_to_term[i], round(float(row[i]), 4))
                for i in top_idx if row[i] > 0
            ])
        return results

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
    def recall_at_k(
        relevant: list[int], retrieved: list[int], k: int = 5
    ) -> float:
        if not relevant or not retrieved:
            return 0.0
        retrieved_k = retrieved[:k]
        hits = len(set(relevant) & set(retrieved_k))
        return hits / len(relevant)

    @staticmethod
    def mean_reciprocal_rank(
        query_relevant: list[list[int]],
        query_retrieved: list[list[int]],
    ) -> float:
        rrs = []
        for rel, ret in zip(query_relevant, query_retrieved):
            for rank, doc in enumerate(ret, start=1):
                if doc in rel:
                    rrs.append(1.0 / rank)
                    break
            else:
                rrs.append(0.0)
        return float(np.mean(rrs))

    @staticmethod
    def ndcg_at_k(
        relevant: list[int], retrieved: list[int], k: int = 5
    ) -> float:
        if not retrieved or not relevant:
            return 0.0
        retrieved_k = retrieved[:k]
        dcg = 0.0
        for i, doc in enumerate(retrieved_k, start=1):
            if doc in relevant:
                dcg += 1.0 / np.log2(i + 1)
        ideal = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal + 1))
        return dcg / idcg if idcg > 0 else 0.0

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
