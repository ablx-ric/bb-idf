"""Supplementary analysis: similarity of each algorithm's keyword ranking to
TextRank (the reference).

This measures *convergence of rankings*, NOT quality. Quality is evaluated
against the author keywords (see ``run.py``); here we only quantify how similar
each algorithm's top-K list is to TextRank's top-K list, per document.

Metrics (per document, per K):
  * RBO@K  — Rank-Biased Overlap (Webber et al. 2010), truncated, p=0.9.
  * Jaccard@K — |A∩B| / |A∪B| over the top-K sets.
  * Overlap@K — |A∩B| / K.

Reads ``results/metrics/keywords_ranked.csv`` (already computed) and writes:
  * results/processed/similarity_to_textrank.csv
  * results/statistical/similarity_tests.csv   (paired Wilcoxon: bbidf vs tfidf)
  * results/figures/similarity_to_textrank.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from bb_idf.experiment import stats as stats_module

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
KS = [5, 10, 20, 50]
P = 0.9  # RBO persistence


def rbo_at_k(a: list[str], b: list[str], k: int, p: float = P) -> float:
    """Rank-Biased Overlap at depth k (Webber et al. 2010), truncated and
    renormalized so that identical lists give 1.0.

    RBO@k = sum_{d=1..k} p^(d-1) * |S_{:d} ∩ T_{:d}| / d
            divided by sum_{d=1..k} p^(d-1)

    This is the renormalized truncated RBO (not the extrapolated infinite
    version): it ranks lists correctly but its range is [0, 1] only after the
    denominator normalization below.
    """
    set_a: set[str] = set()
    set_b: set[str] = set()
    acc = 0.0
    for d in range(1, k + 1):
        if d <= len(a):
            set_a.add(a[d - 1])
        if d <= len(b):
            set_b.add(b[d - 1])
        acc += (p ** (d - 1)) * (len(set_a & set_b) / d)
    denom = (1.0 - p ** k) / (1.0 - p) if p < 1.0 else float(k)
    return acc / denom if denom > 0 else 0.0


def jaccard_at_k(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def overlap_at_k(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / k if k else 0.0


METRICS = {"rbo": rbo_at_k, "jaccard": jaccard_at_k, "overlap": overlap_at_k}


def run_similarity(resdir="results"):
    resdir = Path(resdir)
    kw = pl.read_csv(resdir / "metrics" / "keywords_ranked.csv")

    # Build ordered top-50 lists per document per algorithm.
    lists: dict[int, dict[str, list[str]]] = {}
    for row in kw.iter_rows(named=True):
        d = row["doc_id"]
        lists.setdefault(d, {}).setdefault(row["algorithm"], []).append(row["term"])

    rows = []
    for d in sorted(lists):
        algos = lists[d]
        if "textrank" not in algos:
            continue
        tr = algos["textrank"]
        for algo in ["tfidf", "bbidf"]:
            a = algos[algo]
            for k in KS:
                for metric, fn in METRICS.items():
                    rows.append({
                        "doc_id": d,
                        "doc": f"doc{d + 1}",
                        "k": k,
                        "metric": metric,
                        "algo": algo,
                        "similarity": round(fn(a, tr, k), 6),
                    })
    df = pl.DataFrame(rows)
    (resdir / "processed").mkdir(parents=True, exist_ok=True)
    df.write_csv(resdir / "processed" / "similarity_to_textrank.csv")

    # Paired comparison: is sim(bbidf, textrank) > sim(tfidf, textrank)?
    stat_rows = []
    for metric in METRICS:
        for k in KS:
            sub = df.filter((pl.col("metric") == metric) & (pl.col("k") == k))
            tf = sub.filter(pl.col("algo") == "tfidf").sort("doc_id")["similarity"].to_numpy()
            bb = sub.filter(pl.col("algo") == "bbidf").sort("doc_id")["similarity"].to_numpy()
            w = stats_module.paired_wilcoxon(tf, bb)
            ci = stats_module.bootstrap_mean_diff_ci(tf, bb, seed=0)
            stat_rows.append({
                "metric": metric, "k": k,
                "mean_tfidf": float(tf.mean()),
                "mean_bbidf": float(bb.mean()),
                "p_value": w["p_value"], "n_pairs": w["n_pairs"],
                "cohens_d": stats_module.cohens_d_paired(tf, bb),
                "mean_diff": ci["mean_diff"],
                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            })
    stat = pl.DataFrame(stat_rows)
    (resdir / "statistical").mkdir(parents=True, exist_ok=True)
    stat.write_csv(resdir / "statistical" / "similarity_tests.csv")

    # Figure: RBO@K distribution, bbidf vs tfidf.
    _plot_rbo(resdir / "processed" / "similarity_to_textrank.csv",
              resdir / "figures")

    return df, stat


def _plot_rbo(csv_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(csv_path).filter(pl.col("metric") == "rbo")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(KS))
    width = 0.35
    for i, (algo, color) in enumerate([("tfidf", OKABE_ITO[0]), ("bbidf", OKABE_ITO[1])]):
        means, errs = [], []
        for k in KS:
            v = df.filter((pl.col("algo") == algo) & (pl.col("k") == k))["similarity"].to_numpy()
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        bars = ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=3,
                      label=algo.upper(), color=color, edgecolor="black", linewidth=0.4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in KS])
    ax.set_ylabel("RBO@K con TextRank (media ± IC 95%)")
    ax.set_title("Similitud de ranking con TextRank (RBO, p=0.9)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(outdir / "similarity_to_textrank.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  OK similarity_to_textrank.png")


if __name__ == "__main__":
    run_similarity()
