import time
import numpy as np
import polars as pl

from tf_inst.preprocessing import Preprocessor
from tf_inst.evaluation.metrics import Evaluator


class Benchmark:
    def __init__(self, preprocessor: Preprocessor | None = None):
        self.preprocessor = preprocessor or Preprocessor()

    def run(
        self,
        documents: list[str],
        queries: list[str],
        vectorizers: dict[str, object],
    ) -> pl.DataFrame:
        evaluator = Evaluator()
        rows = []

        docs_processed = [self.preprocessor(d) for d in documents]
        queries_processed = [self.preprocessor(q) for q in queries]

        for name, vec in vectorizers.items():
            t0 = time.perf_counter()
            D = vec.fit_transform(docs_processed)
            Q = vec.transform(queries_processed)
            fit_time = time.perf_counter() - t0

            sim_matrix = evaluator.cosine_similarity_matrix(Q, D)

            rows.append({
                "algorithm": name,
                "vocab_size": len(vec.vocabulary),
                "matrix_shape": f"{D.shape}",
                "fit_time_s": round(fit_time, 4),
            })

        return pl.DataFrame(rows)
