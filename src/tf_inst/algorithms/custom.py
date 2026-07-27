import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class TFPDC_Scalable:
    def __init__(
        self,
        min_threshold: float = 0.0,
        max_threshold: float = 100.0,
        normalize_aof: bool = False,
        use_pdc: bool = True,
        stop_words=None,
    ):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.normalize_aof = normalize_aof
        self.use_pdc = use_pdc
        self.vectorizer = CountVectorizer(stop_words=stop_words)
        self._vocabulary = None
        self._valid_mask = None
        self._idf = None
        self._inv_FTC = None
        self.AOF = None
        self.PDC_doc = None
        self.FTD = None
        self.FTC = None

    def fit(self, documents: list[str]):
        X_sparse = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        n_docs = X_sparse.shape[0]

        if self.normalize_aof:
            doc_lengths = np.asarray(X_sparse.sum(axis=1)).flatten()
            doc_lengths[doc_lengths == 0] = 1
            X_normalized = X_sparse.multiply(1 / doc_lengths[:, np.newaxis])
            term_freq_sum = np.asarray(X_normalized.sum(axis=0)).flatten()
        else:
            term_freq_sum = np.asarray(X_sparse.sum(axis=0)).flatten()
        self.AOF = term_freq_sum / n_docs

        self._valid_mask = (self.AOF >= self.min_threshold) & (
            self.AOF <= self.max_threshold
        )
        X_filtered = X_sparse[:, self._valid_mask].tocsr()
        terms = feature_names[self._valid_mask]
        self._vocabulary = {term: idx for idx, term in enumerate(terms)}

        if X_filtered.shape[1] == 0:
            raise ValueError("El filtro AOF eliminó todos los términos. Revisa min_threshold/max_threshold y normalize_aof.")

        FTD = np.asarray(X_filtered.sum(axis=1)).flatten()
        FTC = np.asarray(X_filtered.sum(axis=0)).flatten()
        total_collection_terms = FTC.sum()
        self.FTD = FTD
        self.FTC = FTC
        self.PDC_doc = (
            FTD / total_collection_terms
            if total_collection_terms > 0
            else np.zeros_like(FTD, dtype=float)
        )

        doc_freq = np.asarray((X_filtered > 0).sum(axis=0)).flatten()
        self._idf = np.log(n_docs / (doc_freq + 1e-8))

        inv_FTC = np.zeros_like(FTC, dtype=float)
        nonzero = FTC > 0
        inv_FTC[nonzero] = 1.0 / FTC[nonzero]
        self._inv_FTC = inv_FTC

        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self._idf is None:
            raise ValueError("El vectorizador no ha sido ajustado. Llama a fit() primero.")
        X = self.vectorizer.transform(documents)[:, self._valid_mask].tocsr()

        if not self.use_pdc:
            return np.asarray(X.multiply(self._idf).toarray())

        pdc_term_doc = X.multiply(self._inv_FTC).tocsr()
        idf_broadcast = sparse.csr_matrix(self._idf.reshape(1, -1))
        score = pdc_term_doc.multiply(idf_broadcast)
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
