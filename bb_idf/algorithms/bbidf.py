import math

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BBIDF:

    def __init__(self, stop_words=None):
        self.vectorizer = CountVectorizer(stop_words=stop_words)
        self._vocabulary = None
        self._idf = None
        self._df_banda = None

    def fit(self, documents: list[str]):
        X = self.vectorizer.fit_transform(documents).toarray()
        feature_names = self.vectorizer.get_feature_names_out()
        n_docs, n_terms = X.shape

        band_inf, band_sup = self._compute_bands(X)

        self._df_banda = np.zeros(n_terms)
        for t in range(n_terms):
            col = X[:, t]
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
            raise ValueError(
                "El vectorizador no ha sido ajustado. Llama a fit() primero."
            )

        X = self.vectorizer.transform(documents).toarray()
        n_docs, n_terms = X.shape

        band_inf, band_sup = self._compute_bands(X)

        for d in range(n_docs):
            for t in range(n_terms):
                tf_abs = X[d, t]
                if tf_abs > 0 and not (band_inf[d] <= tf_abs <= band_sup[d]):
                    X[d, t] = 0.0

        return X * self._idf

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    @staticmethod
    def _compute_bands(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_docs = X.shape[0]
        token_counts = np.asarray(X.sum(axis=1)).flatten()
        band_inf = np.zeros(n_docs)
        band_sup = np.zeros(n_docs)

        for d in range(n_docs):
            freqs = X[d, :]
            nonzero_freqs = freqs[freqs > 0]
            if len(nonzero_freqs) > 0:
                mean_f = np.mean(nonzero_freqs)
                std_f = np.std(nonzero_freqs)
                band_inf[d] = mean_f + 0.5 * std_f
                band_sup[d] = mean_f + 2.5 * std_f
            if band_inf[d] >= band_sup[d] or token_counts[d] < 30:
                band_inf[d] = 1.5
                band_sup[d] = 4.5

        return band_inf, band_sup

    @property
    def vocabulary(self) -> dict:
        return self._vocabulary

    @property
    def feature_names(self) -> list[str]:
        return sorted(self._vocabulary, key=self._vocabulary.get)
