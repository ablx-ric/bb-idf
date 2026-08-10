import math

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BBIDF:
    def __init__(self, stop_words=None):
        self.vectorizer = CountVectorizer(stop_words=stop_words)
        self._vocabulary = None
        self._idf = None
        self._df_banda = None

    def fit(self, documents: list[str]):
        X_sparse = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        n_docs = X_sparse.shape[0]
        n_terms = X_sparse.shape[1]

        X_dense = X_sparse.toarray()
        token_counts = np.asarray(X_dense.sum(axis=1)).flatten()

        band_inf = np.zeros(n_docs)
        band_sup = np.zeros(n_docs)
        for d in range(n_docs):
            freqs = X_dense[d, :]
            nonzero_freqs = freqs[freqs > 0]
            if len(nonzero_freqs) > 0:
                mean_f = np.mean(nonzero_freqs)
                std_f = np.std(nonzero_freqs)
                band_inf[d] = mean_f + 0.5 * std_f
                band_sup[d] = mean_f + 2.5 * std_f
            if band_inf[d] >= band_sup[d] or token_counts[d] < 30:
                band_inf[d] = 1.5
                band_sup[d] = 4.5

        self._df_banda = np.zeros(n_terms)
        for t in range(n_terms):
            col = X_dense[:, t]
            for d in range(n_docs):
                tf_abs = col[d]
                if band_inf[d] <= tf_abs <= band_sup[d]:
                    self._df_banda[t] += 1

        self._idf = np.array([
            math.log(1 + n_docs / (df_val + 1))
            for df_val in self._df_banda
        ])
        self._vocabulary = {term: idx for idx, term in enumerate(feature_names)}

        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self._idf is None:
            raise ValueError("El vectorizador no ha sido ajustado. Llama a fit() primero.")
        X = self.vectorizer.transform(documents).tocsr()
        idf_broadcast = sparse.csr_matrix(self._idf.reshape(1, -1))
        score = X.multiply(idf_broadcast)
        return np.asarray(score.toarray())

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    def top_keywords(self, documents: list[str], top_n: int = 10) -> list[list[tuple]]:
        matriz = self.transform(documents)
        resultados = []
        for fila in matriz:
            top_idx = fila.argsort()[::-1][:top_n]
            resultados.append(
                [(self.feature_names[i], round(float(fila[i]), 4)) for i in top_idx if fila[i] > 0]
            )
        return resultados

    @property
    def vocabulary(self) -> dict:
        return self._vocabulary

    @property
    def feature_names(self) -> list[str]:
        return sorted(self._vocabulary, key=self._vocabulary.get)
