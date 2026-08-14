"""BB-IDF (Bounded Band Inverse Document Frequency) — definición validada.

Definición (idéntica a ``bb_idf.experiment.scorers.bbidf_weights``, que es la
usada por el experimento oficial en ``run_all.py``):

    w(t, d) = tf(t, d) * idf_banda(t)
    idf_banda(t) = ln((1 + N) / (1 + df_banda(t))) + 1
    df_banda(t)  = nº de documentos donde la frecuencia f(t, d) cae dentro de
                   la banda estadística del documento d:
                   [mu_d + 0.5*sigma_d, mu_d + 2.5*sigma_d]
                   (mu/sigma sobre las frecuencias NO nulas del documento;
                   banda de respaldo [1.5, 4.5] si la banda es degenerada o el
                   documento tiene menos de 30 tokens).

La ÚNICA diferencia respecto a TF-IDF es ``df -> df_banda``: misma fórmula de
IDF suavizado, mismo TF crudo y SIN anulación de pesos fuera de banda.

NOTA HISTÓRICA: una versión anterior de esta clase aplicaba además un filtro
duro (anulaba en ``transform`` los pesos de términos fuera de banda) y usaba
otra fórmula de IDF. Esa variante resultó inviable para extracción de keywords
(F1@10 ≈ 0.04, anula ~91% de los términos en documentos largos; ver
``docs/INFORME_EXPERIMENTAL.md`` y la variante ``bbidf_hard_band`` de
``results/statistical/robustness.csv``). La variante dura se conserva solo
para el análisis de robustez en
``bb_idf/experiment/scorers.py::bbidf_weights_hard``.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Coeficientes de la banda (de la propuesta): inferior = mu + 0.5*sigma,
# superior = mu + 2.5*sigma; banda de respaldo para docs cortos/degenerados.
BAND_LO_COEF = 0.5
BAND_HI_COEF = 2.5
BAND_FALLBACK = (1.5, 4.5)
SHORT_DOC_TOKENS = 30


class BBIDF:
    """Vectorizador BB-IDF con la definición validada (sin filtro duro)."""

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

        df_banda = np.zeros(n_terms, dtype=np.float64)
        for d in range(n_docs):
            in_band = (X[d, :] >= band_inf[d]) & (X[d, :] <= band_sup[d])
            df_banda += in_band

        self._df_banda = df_banda
        # Mismo IDF suavizado que el experimento: ln((1+N)/(1+df)) + 1.
        self._idf = np.log((1.0 + n_docs) / (1.0 + df_banda)) + 1.0
        self._vocabulary = {term: idx for idx, term in enumerate(feature_names)}

        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self._idf is None:
            raise ValueError(
                "El vectorizador no ha sido ajustado. Llama a fit() primero."
            )

        X = self.vectorizer.transform(documents).toarray().astype(np.float64)
        # Sin anulación de pesos fuera de banda: solo cambia el IDF.
        return X * self._idf[None, :]

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    @staticmethod
    def _compute_bands(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Banda estadística por documento sobre frecuencias no nulas."""
        n_docs = X.shape[0]
        token_counts = np.asarray(X.sum(axis=1)).flatten()
        band_inf = np.zeros(n_docs)
        band_sup = np.zeros(n_docs)

        for d in range(n_docs):
            freqs = X[d, :]
            nonzero_freqs = freqs[freqs > 0]
            if len(nonzero_freqs) > 0:
                mean_f = np.mean(nonzero_freqs)
                std_f = np.std(nonzero_freqs)  # std poblacional (ddof=0)
                band_inf[d] = mean_f + BAND_LO_COEF * std_f
                band_sup[d] = mean_f + BAND_HI_COEF * std_f
            if band_inf[d] >= band_sup[d] or token_counts[d] < SHORT_DOC_TOKENS:
                band_inf[d], band_sup[d] = BAND_FALLBACK

        return band_inf, band_sup

    @property
    def vocabulary(self) -> dict:
        return self._vocabulary

    @property
    def feature_names(self) -> list[str]:
        return sorted(self._vocabulary, key=self._vocabulary.get)
