"""Keyword-extraction metrics (standard definitions).

Ground truth is a SET of gold unigram terms per document (the author-declared
keywords, normalized). Metrics are computed per document over the ranked
keyword list of each algorithm.
"""

from __future__ import annotations

import numpy as np


def ranked_terms(weight_row: np.ndarray, vocab: list[str],
                 k: int | None = None) -> list[str]:
    """Terms ranked by descending weight; ties broken by ascending term.

    Terms with weight <= 0 are excluded. If ``k`` is given, only the top-k are
    returned.
    """
    order = sorted(
        range(len(vocab)),
        key=lambda i: (-weight_row[i], vocab[i]),
    )
    out = [vocab[i] for i in order if weight_row[i] > 0]
    return out[:k] if k is not None else out


def ranked_terms_with_scores(weight_row: np.ndarray, vocab: list[str],
                             k: int | None = None) -> list[tuple[str, float]]:
    """Like :func:`ranked_terms` but returns ``(term, weight)`` pairs."""
    order = sorted(
        range(len(vocab)),
        key=lambda i: (-weight_row[i], vocab[i]),
    )
    out = [(vocab[i], float(weight_row[i])) for i in order if weight_row[i] > 0]
    return out[:k] if k is not None else out


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    top = ranked[:k]
    return len(set(top) & gold) / k


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = ranked[:k]
    return len(set(top) & gold) / len(gold)


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def average_precision(ranked: list[str], gold: set[str]) -> float:
    """AP over the full ranked list with binary relevance."""
    if not gold or not ranked:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, term in enumerate(ranked, start=1):
        if term in gold:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, term in enumerate(ranked, start=1):
        if term in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """nDCG@k with binary relevance (1 if in gold, else 0)."""
    if not gold or k <= 0 or not ranked:
        return 0.0
    top = ranked[:k]
    dcg = 0.0
    for i, term in enumerate(top, start=1):
        if term in gold:
            dcg += 1.0 / np.log2(i + 1)
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_document(weight_row: np.ndarray, vocab: list[str],
                      gold: set[str], ks: list[int]) -> dict:
    ranked = ranked_terms(weight_row, vocab)
    out: dict = {"n_retrieved": len(ranked), "n_gold": len(gold)}
    for k in ks:
        p = precision_at_k(ranked, gold, k)
        r = recall_at_k(ranked, gold, k)
        out[f"P@{k}"] = p
        out[f"R@{k}"] = r
        out[f"F1@{k}"] = f1_at_k(p, r)
        out[f"nDCG@{k}"] = ndcg_at_k(ranked, gold, k)
    out["AP"] = average_precision(ranked, gold)
    out["MRR"] = reciprocal_rank(ranked, gold)
    return out
