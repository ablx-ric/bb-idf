import numpy as np


class TextRankVectorizer:
    """TextRank (Mihalcea & Tarau, 2004) aplicado a ponderación de términos.

    Para cada documento construye un grafo de co-ocurrencia de términos
    (ventana deslizante) y calcula la relevancia de cada término con
    PageRank (iteración de potencias). Expone la misma interfaz que
    TfidfVectorizerWrapper para ser intercambiable en el benchmark.
    """

    def __init__(
        self,
        window_size: int = 2,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.window_size = window_size
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self._vocabulary = None

    def fit(self, documents: list[str]):
        vocab: dict[str, int] = {}
        for doc in documents:
            for token in doc.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocabulary = vocab
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self._vocabulary is None:
            raise ValueError("El vectorizador no ha sido ajustado. Llama a fit() primero.")
        X = np.zeros((len(documents), len(self._vocabulary)))
        for i, doc in enumerate(documents):
            scores = self._score_document(doc.split())
            for term, score in scores.items():
                j = self._vocabulary.get(term)
                if j is not None:
                    X[i, j] = score
        return X

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    def _score_document(self, tokens: list[str]) -> dict[str, float]:
        terms = list(dict.fromkeys(tokens))
        if not terms:
            return {}
        index = {t: k for k, t in enumerate(terms)}
        n = len(terms)

        W = np.zeros((n, n))
        for pos, token in enumerate(tokens):
            for other in tokens[pos + 1 : pos + 1 + self.window_size]:
                if other != token:
                    a, b = index[token], index[other]
                    W[a, b] += 1.0
                    W[b, a] += 1.0

        out_weight = W.sum(axis=1)
        safe_out = np.where(out_weight > 0, out_weight, 1.0)

        scores = np.ones(n)
        for _ in range(self.max_iter):
            prev = scores
            scores = (1 - self.damping) + self.damping * (W @ (prev / safe_out))
            if np.abs(scores - prev).sum() < self.tol:
                break

        return {t: float(scores[index[t]]) for t in terms}

    @property
    def vocabulary(self) -> dict:
        return self._vocabulary

    @property
    def feature_names(self) -> list[str]:
        return sorted(self._vocabulary, key=self._vocabulary.get)
