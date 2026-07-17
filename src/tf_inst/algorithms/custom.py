import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class TFPDC_Scalable:
    """Propuesta (tf-inst): TF-IDF con filtrado de términos por AOF.

    AOF (Average Occurrence Frequency) filtra los términos cuyo promedio de
    ocurrencia en la colección queda fuera de [min_threshold, max_threshold]
    antes de ponderar con TF*IDF. Calcula además FTD, FTC y PDC.

    min_threshold: Límite inferior para el AOF.
    max_threshold: Límite superior para el AOF.
    normalize_aof: Si es True, divide las frecuencias de cada documento por su
                   longitud total antes de calcular el AOF (queda entre 0 y 1).
    stop_words: Se pasa a CountVectorizer (p.ej. 'english' o lista propia).
                Por defecto None porque el Preprocessor ya limpia el texto.

    Implementa la misma interfaz que TfidfVectorizerWrapper para ser
    intercambiable en el benchmark.
    """

    def __init__(
        self,
        min_threshold: float = 0.0,
        max_threshold: float = 100.0,
        normalize_aof: bool = False,
        stop_words=None,
    ):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.normalize_aof = normalize_aof
        self.vectorizer = CountVectorizer(stop_words=stop_words)
        self._vocabulary = None
        self._valid_mask = None
        self._idf = None
        self.AOF = None
        self.PDC = None

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

        X_filtered = X_sparse[:, self._valid_mask]
        terms = feature_names[self._valid_mask]
        self._vocabulary = {term: idx for idx, term in enumerate(terms)}

        FTD = np.asarray(X_filtered.sum(axis=1)).flatten()
        FTC = np.asarray(X_filtered.sum(axis=0)).flatten()
        total_collection_terms = FTC.sum()
        self.PDC = (
            FTD / total_collection_terms
            if total_collection_terms > 0
            else np.zeros_like(FTD, dtype=float)
        )

        doc_freq = np.asarray((X_filtered > 0).sum(axis=0)).flatten()
        self._idf = np.log(n_docs / (doc_freq + 1e-8))
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self._idf is None:
            raise ValueError("El vectorizador no ha sido ajustado. Llama a fit() primero.")
        X = self.vectorizer.transform(documents)[:, self._valid_mask]
        return np.asarray(X.multiply(self._idf).toarray())

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    @property
    def vocabulary(self) -> dict:
        return self._vocabulary

    @property
    def feature_names(self) -> list[str]:
        return sorted(self._vocabulary, key=self._vocabulary.get)
