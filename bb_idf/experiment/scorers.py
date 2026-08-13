"""Keyword scorers over a shared term-document count matrix.

TF-IDF and BB-IDF use the SAME IDF formula (smooth inverse document frequency,
as in the project's reference notebooks) and the SAME raw TF. BB-IDF differs
from TF-IDF ONLY in how the document frequency is counted: it counts a
document for ``df(t)`` only when the term's frequency falls inside that
document's statistical band (the "Bounded Band" filter). This isolates the
effect of the band filter on the global weighting.
"""

from __future__ import annotations

import numpy as np

# Band coefficients (from the proposal): lower = mu + 0.5*sigma,
# upper = mu + 2.5*sigma; fallback band for short/ degenerate documents.
BAND_LO_COEF = 0.5
BAND_HI_COEF = 2.5
BAND_FALLBACK = (1.5, 4.5)
SHORT_DOC_TOKENS = 30


def _idf_formula(N: int, df: np.ndarray) -> np.ndarray:
    """Smooth IDF shared by TF-IDF and BB-IDF: ln((1+N)/(1+df)) + 1."""
    return np.log((1.0 + N) / (1.0 + df.astype(np.float64))) + 1.0


def _document_frequency(X: np.ndarray) -> np.ndarray:
    """Classic df(t) = number of documents where the term appears at least once."""
    return (X > 0).sum(axis=0).astype(np.float64)


def _compute_bands(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-document statistical band [lo, hi] over nonzero term frequencies."""
    n_docs = X.shape[0]
    token_counts = X.sum(axis=1)
    band_inf = np.zeros(n_docs)
    band_sup = np.zeros(n_docs)
    for d in range(n_docs):
        freqs = X[d, :]
        nz = freqs[freqs > 0].astype(np.float64)
        if nz.size > 0:
            mean_f = nz.mean()
            std_f = nz.std()  # population std (ddof=0), as in the proposal
            band_inf[d] = mean_f + BAND_LO_COEF * std_f
            band_sup[d] = mean_f + BAND_HI_COEF * std_f
        if band_inf[d] >= band_sup[d] or token_counts[d] < SHORT_DOC_TOKENS:
            band_inf[d], band_sup[d] = BAND_FALLBACK
    return band_inf, band_sup


def _band_document_frequency(X: np.ndarray) -> np.ndarray:
    """df_banda(t) = number of docs where f(t,d) falls inside that doc's band."""
    band_inf, band_sup = _compute_bands(X)
    n_docs, n_terms = X.shape
    df_banda = np.zeros(n_terms, dtype=np.float64)
    for d in range(n_docs):
        in_band = (X[d, :] >= band_inf[d]) & (X[d, :] <= band_sup[d])
        df_banda += in_band.astype(np.float64)
    return df_banda


def tfidf_weights(X: np.ndarray) -> np.ndarray:
    """w(t,d) = tf(t,d) * idf(t) with classic df."""
    N = X.shape[0]
    df = _document_frequency(X)
    idf = _idf_formula(N, df)
    return X.astype(np.float64) * idf[None, :]


def bbidf_weights(X: np.ndarray) -> np.ndarray:
    """w(t,d) = tf(t,d) * idf_bb(t), where idf_bb uses df_banda.

    Same IDF formula and same TF as ``tfidf_weights``; only ``df`` is replaced
    by the band-filtered document frequency. (No hard zeroing of out-of-band
    weights: that would be an additional change beyond the band filter.)
    """
    N = X.shape[0]
    df_banda = _band_document_frequency(X)
    idf_bb = _idf_formula(N, df_banda)
    return X.astype(np.float64) * idf_bb[None, :]


def bbidf_weights_hard(X: np.ndarray) -> np.ndarray:
    """BB-IDF variant with the HARD band filter (as in the packaged bbidf.py).

    In addition to using df_banda in the IDF, out-of-band term weights are
    zeroed (``transform`` in ``bb_idf/algorithms/bbidf.py``). Same smooth IDF
    formula as ``bbidf_weights``, so the only difference here is the hard
    zeroing.
    """
    N = X.shape[0]
    df_banda = _band_document_frequency(X)
    idf_bb = _idf_formula(N, df_banda)
    band_inf, band_sup = _compute_bands(X)
    W = X.astype(np.float64).copy()
    for d in range(X.shape[0]):
        in_band = (W[d, :] >= band_inf[d]) & (W[d, :] <= band_sup[d])
        W[d, ~in_band] = 0.0
    return W * idf_bb[None, :]


def _pagerank_document(tokens: list[str], window: int, damping: float,
                       max_iter: int, tol: float) -> dict[str, float]:
    """Weighted PageRank (Mihalcea & Tarau 2004) over a co-occurrence graph."""
    terms = list(dict.fromkeys(tokens))
    if not terms:
        return {}
    index = {t: k for k, t in enumerate(terms)}
    n = len(terms)

    W = np.zeros((n, n))
    for pos, token in enumerate(tokens):
        for other in tokens[pos + 1: pos + 1 + window]:
            if other != token:
                a, b = index[token], index[other]
                W[a, b] += 1.0
                W[b, a] += 1.0

    out_weight = W.sum(axis=1)
    safe_out = np.where(out_weight > 0, out_weight, 1.0)

    scores = np.ones(n)
    for _ in range(max_iter):
        prev = scores
        scores = (1.0 - damping) + damping * (W @ (prev / safe_out))
        if np.abs(scores - prev).sum() < tol:
            break
    return {t: float(scores[index[t]]) for t in terms}


def textrank_weights(docs_tokens: list[list[str]], vocab: list[str],
                     window: int = 2, damping: float = 0.85,
                     max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Per-document TextRank scores aligned to the shared vocabulary."""
    W = np.zeros((len(docs_tokens), len(vocab)))
    vidx = {t: i for i, t in enumerate(vocab)}
    for d, tokens in enumerate(docs_tokens):
        scores = _pagerank_document(tokens, window, damping, max_iter, tol)
        for term, score in scores.items():
            j = vidx.get(term)
            if j is not None:
                W[d, j] = score
    return W


SCORERS = {
    "tfidf": tfidf_weights,
    "bbidf": bbidf_weights,
    "textrank": textrank_weights,
}
