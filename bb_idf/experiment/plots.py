"""Scientific visualizations for the keyword-extraction experiment.

Reads results from ``results/`` (CSV/JSON) and writes figures to
``results/figures/``. All figures are 2D, use a colorblind-safe palette
(Okabe-Ito), annotate values, and follow a clean publication style.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from wordcloud import WordCloud

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
ALGO_COLORS = {"tfidf": "#E69F00", "bbidf": "#56B4E9", "textrank": "#009E73"}
ALGO_LABELS = {"tfidf": "TF-IDF", "bbidf": "BB-IDF", "textrank": "TextRank"}


def _style():
    mpl.rcParams.update({
        "figure.dpi": 100, "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "font.size": 9, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "black", "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "image.cmap": "viridis", "legend.frameon": False,
    })


def _save(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    print(f"  OK {name}.png")


def _annotate_bars(ax, bars, fmt="{:.2f}", fontsize=7, dy=0.0):
    """Write the value on top of each bar."""
    for bar in bars:
        h = bar.get_height()
        if h == 0 or (isinstance(h, float) and np.isnan(h)):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize)


def _load(df_path):
    return pl.read_csv(df_path)


def plot_at_k_curves(resdir, outdir):
    _style()
    df = _load(Path(resdir) / "raw" / "per_doc_metrics.csv")
    ks = [5, 10, 20, 50]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, metric in zip(axes, ["P", "R", "F1"]):
        for algo in ["tfidf", "bbidf", "textrank"]:
            sub = df.filter(pl.col("algorithm") == algo)
            vals = [sub[f"{metric}@{k}"].mean() for k in ks]
            ax.plot(ks, vals, marker="o", ms=4, lw=1.4,
                    color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
            for k, v in zip(ks, vals):
                ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=6.5,
                            color=ALGO_COLORS[algo])
        ax.set_xticks(ks)
        ax.set_xlabel("K")
        ax.set_ylabel(f"{metric}@K")
        ax.set_title(f"{metric}@K (media por documento)")
    axes[0].legend(loc="upper right")
    _save(fig, outdir, "prf_at_k")


def plot_f1_comparison(resdir, outdir):
    _style()
    df = _load(Path(resdir) / "raw" / "per_doc_metrics.csv")
    ks = [5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(ks))
    width = 0.26
    for i, algo in enumerate(["tfidf", "bbidf", "textrank"]):
        sub = df.filter(pl.col("algorithm") == algo)
        vals = [sub[f"F1@{k}"].mean() for k in ks]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=ALGO_LABELS[algo],
                      color=ALGO_COLORS[algo], edgecolor="black", linewidth=0.4)
        _annotate_bars(ax, bars, "{:.3f}", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("F1@K (media)")
    ax.set_ylim(0, 0.62)
    ax.set_title("F1@K por algoritmo")
    ax.legend()
    _save(fig, outdir, "f1_comparison")


def plot_improvement(resdir, outdir):
    _style()
    imp = _load(Path(resdir) / "processed" / "improvement_bbidf_vs_tfidf.csv")
    ks = [5, 10, 20, 50]
    metrics = ["P", "R", "F1"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, metric in zip(axes, metrics):
        means, errs = [], []
        for k in ks:
            sub = imp.filter(pl.col("metric") == f"{metric}@{k}")
            vals = sub["improvement_pct"].to_numpy()
            vals = vals[~np.isnan(vals)]
            means.append(vals.mean())
            errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        bars = ax.bar(np.arange(len(ks)), means, color=OKABE_ITO[1],
                      edgecolor="black", linewidth=0.4, yerr=errs, capsize=3)
        _annotate_bars(ax, bars, "{:+.1f}%", fontsize=6.5)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_ylabel("Mejora BB-IDF vs TF-IDF (%)")
        ax.set_title(f"{metric}@K (media ± SE)")
    _save(fig, outdir, "improvement_bbidf")


def plot_per_doc_boxplot(resdir, outdir):
    _style()
    df = _load(Path(resdir) / "raw" / "per_doc_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, metric in zip(axes, ["F1@10", "AP"]):
        data = [df.filter(pl.col("algorithm") == a)[metric].to_numpy()
                for a in ["tfidf", "bbidf", "textrank"]]
        bp = ax.boxplot(data, tick_labels=[ALGO_LABELS[a] for a in ["tfidf", "bbidf", "textrank"]],
                        patch_artist=True, widths=0.5)
        for patch, a in zip(bp["boxes"], ["tfidf", "bbidf", "textrank"]):
            patch.set_facecolor(ALGO_COLORS[a])
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        # individual points
        for i, a in enumerate(["tfidf", "bbidf", "textrank"]):
            vals = data[i]
            x = np.random.default_rng(i).normal(i + 1, 0.04, len(vals))
            ax.scatter(x, vals, s=8, alpha=0.35, color="black", zorder=3)
        ax.set_ylabel(metric)
        ax.set_title(f"Distribución por documento ({metric})")
    _save(fig, outdir, "per_doc_boxplot")


def plot_efficiency(resdir, outdir):
    _style()
    meta = json.loads((Path(resdir) / "metadata.json").read_text(encoding="utf-8"))
    fit = meta["fit_time_s"]
    pdoc = meta["per_doc_time_s"]
    order = ["tfidf", "bbidf", "textrank"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    bars0 = axes[0].bar(order, [fit[a] for a in order],
                        color=[ALGO_COLORS[a] for a in order],
                        edgecolor="black", linewidth=0.4)
    _annotate_bars(axes[0], bars0, "{:.4f}s", fontsize=6.5)
    axes[0].set_ylabel("Tiempo total (s)")
    axes[0].set_title("Tiempo de ajuste")
    bars1 = axes[1].bar(order, [pdoc[a] for a in order],
                        color=[ALGO_COLORS[a] for a in order],
                        edgecolor="black", linewidth=0.4)
    _annotate_bars(axes[1], bars1, "{:.5f}s", fontsize=6.5)
    axes[1].set_ylabel("Tiempo por documento (s)")
    axes[1].set_title("Costo por documento")
    axes[0].set_xticks(range(len(order)), [ALGO_LABELS[a] for a in order])
    axes[1].set_xticks(range(len(order)), [ALGO_LABELS[a] for a in order])
    _save(fig, outdir, "efficiency")


def plot_quality_time_tradeoff(resdir, outdir):
    _style()
    df = _load(Path(resdir) / "raw" / "per_doc_metrics.csv")
    meta = json.loads((Path(resdir) / "metadata.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for algo in ["tfidf", "bbidf", "textrank"]:
        f1 = df.filter(pl.col("algorithm") == algo)["F1@10"].mean()
        t = meta["fit_time_s"][algo]
        ax.scatter(t, f1, s=120, color=ALGO_COLORS[algo], edgecolor="black",
                   zorder=3, label=ALGO_LABELS[algo])
        ax.annotate(f"{ALGO_LABELS[algo]}\nF1={f1:.3f}, t={t:.4f}s", (t, f1),
                    textcoords="offset points", xytext=(8, 8), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de ajuste (s, escala log)")
    ax.set_ylabel("F1@10 (media)")
    ax.legend()
    _save(fig, outdir, "quality_time_tradeoff")


def plot_heatmap(resdir, outdir):
    _style()
    df = _load(Path(resdir) / "raw" / "per_doc_metrics.csv")
    algos = ["tfidf", "bbidf", "textrank"]
    docs = sorted(df["doc_id"].unique().to_list())
    M = np.zeros((len(algos), len(docs)))
    for i, a in enumerate(algos):
        sub = df.filter(pl.col("algorithm") == a).sort("doc_id")
        M[i] = sub["F1@10"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([ALGO_LABELS[a] for a in algos])
    ax.set_xticks(range(len(docs)))
    ax.set_xticklabels([str(d) for d in docs], fontsize=6)
    ax.set_xlabel("Documento")
    ax.set_title("F1@10 por documento y algoritmo")
    for i in range(len(algos)):
        for j in range(len(docs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if M[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, label="F1@10")
    _save(fig, outdir, "heatmap_f1_doc")


def plot_ci(resdir, outdir):
    _style()
    df = _load(Path(resdir) / "raw" / "per_doc_metrics.csv")
    ks = [5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(ks))
    width = 0.26
    for i, algo in enumerate(["tfidf", "bbidf", "textrank"]):
        sub = df.filter(pl.col("algorithm") == algo)
        means, errs = [], []
        for k in ks:
            v = sub[f"F1@{k}"].to_numpy()
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3,
                      label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
                      edgecolor="black", linewidth=0.4)
        _annotate_bars(ax, bars, "{:.3f}", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("F1@K (media ± IC 95%)")
    ax.set_ylim(0, 0.62)
    ax.set_title("F1@K con intervalo de confianza 95%")
    ax.legend()
    _save(fig, outdir, "f1_with_ci")


def plot_paired_diff(resdir, outdir):
    """Distribution of per-document F1@10 difference (BB-IDF - TF-IDF)."""
    _style()
    df = _load(Path(resdir) / "raw" / "per_doc_metrics.csv")
    tf = df.filter(pl.col("algorithm") == "tfidf").sort("doc_id")["F1@10"].to_numpy()
    bb = df.filter(pl.col("algorithm") == "bbidf").sort("doc_id")["F1@10"].to_numpy()
    diff = bb - tf
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(diff, bins=15, color=OKABE_ITO[1], edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(diff.mean(), color="#D55E00", lw=1.2, ls="--",
               label=f"media={diff.mean():+.3f}")
    ax.set_xlabel("F1@10 (BB-IDF − TF-IDF)")
    ax.set_ylabel("Nº documentos")
    ax.set_title(f"n={len(diff)} docs; >0 en {(diff > 0).sum()}, <0 en {(diff < 0).sum()}, "
                 f"=0 en {(diff == 0).sum()}")
    ax.legend()
    _save(fig, outdir, "paired_diff_f1")


def plot_wordclouds(resdir, outdir):
    """Word clouds of the keywords produced by each algorithm (score-weighted)."""
    kw = _load(Path(resdir) / "metrics" / "keywords_ranked.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, algo in zip(axes, ["tfidf", "bbidf", "textrank"]):
        sub = kw.filter(pl.col("algorithm") == algo)
        term_scores: dict[str, float] = {}
        for term, score in zip(sub["term"].to_list(), sub["score"].to_list()):
            term_scores[term] = term_scores.get(term, 0.0) + score
        wc = WordCloud(width=500, height=320, background_color="white",
                       colormap="viridis", max_words=60,
                       relative_scaling=0.5,
                       random_state=0).generate_from_frequencies(term_scores)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{ALGO_LABELS[algo]}", fontsize=12)
    fig.suptitle("Nube de palabras clave por algoritmo (score acumulado en Top-50)", fontsize=13)
    _save(fig, outdir, "wordclouds")


def plot_keyword_frequency(resdir, outdir):
    """Top-20 terms most often in each algorithm's Top-10 across documents."""
    _style()
    kw = _load(Path(resdir) / "metrics" / "keywords_ranked.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, algo in zip(axes, ["tfidf", "bbidf", "textrank"]):
        sub = kw.filter(pl.col("algorithm") == algo)
        counter = Counter()
        for term in sub["term"].to_list():
            counter[term] += 1
        top = counter.most_common(20)
        terms = [t for t, _ in top][::-1]
        counts = [c for _, c in top][::-1]
        ax.barh(range(len(terms)), counts, color=ALGO_COLORS[algo],
                edgecolor="black", linewidth=0.3)
        for i, c in enumerate(counts):
            ax.text(c, i, f" {c}", va="center", fontsize=6.5)
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms, fontsize=7)
        ax.set_xlabel("Docs en Top-10")
        ax.set_title(ALGO_LABELS[algo])
    fig.suptitle("Frecuencia de keywords (top-20) en el Top-10", fontsize=12)
    _save(fig, outdir, "keyword_frequency")


def plot_all(resdir="results", outdir=None):
    outdir = outdir or (Path(resdir) / "figures")
    _style()
    print("Generando figuras...")
    plot_at_k_curves(resdir, outdir)
    plot_f1_comparison(resdir, outdir)
    plot_improvement(resdir, outdir)
    plot_per_doc_boxplot(resdir, outdir)
    plot_efficiency(resdir, outdir)
    plot_quality_time_tradeoff(resdir, outdir)
    plot_heatmap(resdir, outdir)
    plot_ci(resdir, outdir)
    plot_paired_diff(resdir, outdir)
    plot_wordclouds(resdir, outdir)
    plot_keyword_frequency(resdir, outdir)
    print(f"  Figuras guardadas en {outdir}/")


if __name__ == "__main__":
    plot_all()
