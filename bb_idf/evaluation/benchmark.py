import pickle
import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from bb_idf.preprocessing import Preprocessor
from bb_idf.evaluation.metrics import Evaluator


@dataclass
class AlgoMetrics:
    name: str
    vocab_size: int
    matrix_shape: tuple
    fit_time: float
    transform_time: float
    query_time: float
    sparsity: float
    density: float
    matrix_memory_kb: float
    serialized_size_kb: float
    sim_matrix: np.ndarray
    precision_scores: dict[int, float]
    recall_scores: dict[int, float]
    ndcg_scores: dict[int, float]
    map_score: float
    mrr_score: float


@dataclass
class BenchmarkResult:
    name: str
    n_docs: int
    n_queries: int
    metrics: list[AlgoMetrics]


class Benchmark:
    def __init__(self, preprocessor: Preprocessor | None = None):
        self.preprocessor = preprocessor or Preprocessor()
        self.result: BenchmarkResult | None = None

    def run(
        self,
        documents: list[str],
        queries: list[str],
        vectorizers: dict[str, object],
    ) -> pl.DataFrame:
        evaluator = Evaluator()
        rows = []
        all_metrics: list[AlgoMetrics] = []

        docs_processed = [self.preprocessor(d) for d in documents]
        queries_processed = [self.preprocessor(q) for q in queries]

        self._last_docs_processed = docs_processed

        n_queries = len(queries_processed)
        query_relevant = [[i] for i in range(min(n_queries, len(docs_processed)))]

        for name, vec in vectorizers.items():
            t0 = time.perf_counter()
            D = vec.fit_transform(docs_processed)
            fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            Q = vec.transform(queries_processed)
            transform_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            sim_matrix = evaluator.cosine_similarity_matrix(Q, D)
            query_retrieved = [
                np.argsort(-sim_matrix[i]).tolist()
                for i in range(n_queries)
            ]
            query_time = time.perf_counter() - t0

            vocab_size = len(vec.vocabulary)
            n_rows, n_cols = D.shape
            nnz = np.count_nonzero(D)
            total_cells = n_rows * n_cols
            sparsity = 1.0 - (nnz / total_cells) if total_cells > 0 else 0.0
            density = nnz / total_cells if total_cells > 0 else 0.0

            matrix_memory = D.nbytes / 1024.0 if hasattr(D, 'nbytes') else (nnz * 8) / 1024.0

            serialized = pickle.dumps(vec)
            serialized_size = len(serialized) / 1024.0

            precision_scores = {}
            recall_scores = {}
            ndcg_scores = {}
            for k in [1, 3, 5, 10]:
                precs = [
                    evaluator.precision_at_k(rel, ret, k=k)
                    for rel, ret in zip(query_relevant, query_retrieved)
                ]
                recs = [
                    evaluator.recall_at_k(rel, ret, k=k)
                    for rel, ret in zip(query_relevant, query_retrieved)
                ]
                ndcgs = [
                    evaluator.ndcg_at_k(rel, ret, k=k)
                    for rel, ret in zip(query_relevant, query_retrieved)
                ]
                precision_scores[k] = round(float(np.mean(precs)), 4)
                recall_scores[k] = round(float(np.mean(recs)), 4)
                ndcg_scores[k] = round(float(np.mean(ndcgs)), 4)

            map_score = round(
                evaluator.mean_average_precision(query_relevant, query_retrieved), 4
            )
            mrr_score = round(
                evaluator.mean_reciprocal_rank(query_relevant, query_retrieved), 4
            )

            m = AlgoMetrics(
                name=name,
                vocab_size=vocab_size,
                matrix_shape=D.shape,
                fit_time=round(fit_time, 4),
                transform_time=round(transform_time, 4),
                query_time=round(query_time, 4),
                sparsity=round(sparsity, 6),
                density=round(density, 6),
                matrix_memory_kb=round(matrix_memory, 2),
                serialized_size_kb=round(serialized_size, 2),
                sim_matrix=sim_matrix,
                precision_scores=precision_scores,
                recall_scores=recall_scores,
                ndcg_scores=ndcg_scores,
                map_score=map_score,
                mrr_score=mrr_score,
            )
            all_metrics.append(m)

            row = {
                "algorithm": name,
                "vocab_size": vocab_size,
                "matrix_shape": f"{D.shape}",
                "fit_time_s": round(fit_time, 4),
                "transform_time_s": round(transform_time, 4),
                "query_time_s": round(query_time, 4),
                "sparsity": round(sparsity, 6),
                "density": round(density, 6),
                "matrix_memory_kb": round(matrix_memory, 2),
                "serialized_size_kb": round(serialized_size, 2),
            }
            for k in [1, 3, 5, 10]:
                row[f"precision@{k}"] = precision_scores[k]
                row[f"recall@{k}"] = recall_scores[k]
                row[f"ndcg@{k}"] = ndcg_scores[k]
            row["MAP"] = map_score
            row["MRR"] = mrr_score
            rows.append(row)

        self.result = BenchmarkResult(
            name="benchmark", n_docs=len(documents), n_queries=n_queries,
            metrics=all_metrics,
        )
        return pl.DataFrame(rows)
