"""Per-document export: top-N keywords (CSV) and word clouds, one folder per
document (``docX``), with L2-normalized scores comparable across algorithms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from wordcloud import WordCloud

from bb_idf.experiment import metrics as metrics_module
from bb_idf.experiment import similarity as similarity_module

ALGORITHMS = ["tfidf", "bbidf", "textrank"]
ALGO_LABELS = {"tfidf": "TF-IDF", "bbidf": "BB-IDF", "textrank": "TextRank"}
# Distinct perceptual colormaps per algorithm for the word clouds.
ALGO_CMAP = {"tfidf": "viridis", "bbidf": "plasma", "textrank": "cividis"}


def _l2_normalized(row: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(row))
    if norm > 0:
        return row / norm
    return row


def export_per_document(documents, docs_tokens, vocab, weights, gold_sets,
                        excluded, output_dir="results/per_document", top_n=50,
                        ks=(5, 10, 20, 50), per_doc_times=None):
    """Write one folder per document with per-algorithm top-N keywords (CSV),
    per-algorithm word clouds, per-document similarity vs TextRank (RBO/Jaccard/
    Overlap), and per-document scoring time, plus a documents index mapping
    ``docX`` to the original filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vidx = {t: i for i, t in enumerate(vocab)}

    index_rows = []
    for d in range(len(documents)):
        if d in excluded:
            continue
        doc_label = f"doc{d + 1}"
        folder = output_dir / doc_label
        folder.mkdir(parents=True, exist_ok=True)
        gold = gold_sets[d]
        index_rows.append({
            "doc": doc_label,
            "index": d,
            "file": documents[d][0],
            "n_tokens": len(docs_tokens[d]),
            "n_gold": len(gold),
            "gold": ", ".join(sorted(gold)),
        })

        ranked_by_algo: dict[str, list[tuple[str, float]]] = {}
        for algo in ALGORITHMS:
            W = weights[algo][d]
            norm_row = _l2_normalized(W)
            ranked = metrics_module.ranked_terms_with_scores(W, vocab, k=top_n)
            ranked_by_algo[algo] = ranked

            rows = []
            freqs: dict[str, float] = {}
            for rank, (term, raw) in enumerate(ranked, start=1):
                rows.append({
                    "rank": rank,
                    "term": term,
                    "score": round(float(norm_row[vidx[term]]), 6),
                    "score_raw": round(raw, 6),
                    "in_gold": term in gold,
                })
                freqs[term] = float(norm_row[vidx[term]])
            pl.DataFrame(rows).write_csv(folder / f"{algo}.csv")

            wc = WordCloud(
                width=800, height=500, background_color="white",
                colormap=ALGO_CMAP[algo], max_words=top_n,
                relative_scaling=0.5, random_state=0,
            ).generate_from_frequencies(freqs)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{doc_label} — {ALGO_LABELS[algo]} (score L2)", fontsize=11)
            fig.savefig(folder / f"wordcloud_{algo}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # Per-document similarity vs TextRank (tfidf / bbidf).
        if "textrank" in ranked_by_algo:
            tr_terms = [t for t, _ in ranked_by_algo["textrank"]]
            sim_rows = []
            for algo in ["tfidf", "bbidf"]:
                a_terms = [t for t, _ in ranked_by_algo[algo]]
                for k in ks:
                    sim_rows.append({
                        "k": k,
                        "algoritmo": algo,
                        "rbo": round(similarity_module.rbo_at_k(a_terms, tr_terms, k), 6),
                        "jaccard": round(similarity_module.jaccard_at_k(a_terms, tr_terms, k), 6),
                        "overlap": round(similarity_module.overlap_at_k(a_terms, tr_terms, k), 6),
                    })
            pl.DataFrame(sim_rows).write_csv(folder / "similarity_textrank.csv")

        # Per-document scoring time.
        if per_doc_times is not None:
            timing_rows = [
                {"algoritmo": algo, "time_s": round(per_doc_times[algo][d], 9)}
                for algo in ALGORITHMS
            ]
            pl.DataFrame(timing_rows).write_csv(folder / "timing.csv")

    pl.DataFrame(index_rows).write_csv(output_dir / "documents_index.csv")
    print(f"  Export por documento guardado en {output_dir}/")
