"""Paired statistical comparisons and bootstrap confidence intervals.

The statistical unit is the DOCUMENT: for a given metric and K, each algorithm
produces one value per document, so comparisons are paired across algorithms
(same documents). We use non-parametric, paired methods as the default.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on b - a."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    mask = diff != 0
    n = int(mask.sum())
    if n == 0:
        return {"n_pairs": n, "statistic": np.nan, "p_value": 1.0,
                "z": np.nan, "note": "no difference"}
    res = stats.wilcoxon(diff[mask], alternative="two-sided")
    return {"n_pairs": n, "statistic": float(res.statistic),
            "p_value": float(res.pvalue)}


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(diff.mean() / sd)


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    t_pos = ranks[nz > 0].sum()
    t_neg = ranks[nz < 0].sum()
    total = t_pos + t_neg
    if total == 0:
        return 0.0
    return float((t_pos - t_neg) / total)


def bootstrap_mean_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                           seed: int = 0) -> dict:
    """95% BCa-like percentile CI for the mean of (b - a)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    n = diff.size
    boots = np.array([
        diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)
    ])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_boot": n_boot,
    }


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
    }
