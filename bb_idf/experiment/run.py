"""Experiment orchestrator: keyword extraction evaluation.

Compares TF-IDF vs BB-IDF vs TextRank on author-declared keywords over the
33-document Spanish tourism corpus.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from bb_idf.experiment import gold as gold_module
from bb_idf.experiment import metrics as metrics_module
from bb_idf.experiment import pipeline as pipeline_module
from bb_idf.experiment import scorers as scorers_module
from bb_idf.experiment import stats as stats_module

KS = [5, 10, 20, 50]
ALGORITHMS = ["tfidf", "bbidf", "textrank"]
ALGO_LABELS = {"tfidf": "TF-IDF", "bbidf": "BB-IDF", "textrank": "TextRank"}


def run_experiment(corpus_dir: str | Path = "data/corpus",
                   output_dir: str | Path = "results",
                   ks: list[int] = KS,
                   textrank_window: int = 2,
                   seed: int = 0) -> dict:
    output_dir = Path(output_dir)
    for sub in ["raw", "processed", "metrics", "statistical", "figures", "tables"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    prep = pipeline_module.preprocess_corpus(corpus_dir)
    documents = prep.documents
    docs_tokens = prep.docs_tokens
    vocab = prep.vocab
    X = prep.X
    gold_sets = prep.gold_sets
    excluded = prep.excluded
    t_prep = time.perf_counter() - t0

    # ---- Validate gold-standard alignment with the corpus ----
    gb = gold_module.gold_by_file()
    corpus_files = {f for f, _ in documents}
    missing_gold = sorted(corpus_files - set(gb))
    unused_gold = sorted(set(gb) - corpus_files)
    if missing_gold:
        print("WARN: documentos del corpus sin entrada en gold:", missing_gold)
    if unused_gold:
        print("WARN: entradas de gold sin archivo en el corpus:", unused_gold)

    n_docs = len(docs_tokens)
    n_gold_docs = n_docs - len(excluded)

    # ---- Scorers (timed) ----
    weights = {}
    times = {}
    for name in ALGORITHMS:
        t = time.perf_counter()
        if name == "textrank":
            W = scorers_module.textrank_weights(
                docs_tokens, vocab, window=textrank_window)
        elif name == "bbidf":
            W = scorers_module.bbidf_weights(X)
        else:
            W = scorers_module.tfidf_weights(X)
        times[name] = time.perf_counter() - t
        weights[name] = W

    # ---- Per-document metrics ----
    rows = []
    keyword_rows = []
    doc_rows = []
    for d in range(n_docs):
        if d in excluded:
            continue
        gold = gold_sets[d]
        doc_rows.append({
            "doc_id": d,
            "file": documents[d][0],
            "n_tokens": len(docs_tokens[d]),
            "n_gold": len(gold),
        })
        for name in ALGORITHMS:
            ev = metrics_module.evaluate_document(
                weights[name][d], vocab, gold, ks)
            row = {"doc_id": d, "file": documents[d][0], "algorithm": name,
                   "n_tokens": len(docs_tokens[d]), "n_gold": len(gold)}
            row.update(ev)
            rows.append(row)
            ranked = metrics_module.ranked_terms(weights[name][d], vocab)
            for rank, term in enumerate(ranked[:max(ks)], start=1):
                keyword_rows.append({
                    "doc_id": d, "algorithm": name, "rank": rank, "term": term,
                    "in_gold": term in gold,
                })

    df = pl.DataFrame(rows)
    df.write_csv(output_dir / "raw" / "per_doc_metrics.csv")

    kw_df = pl.DataFrame(keyword_rows)
    kw_df.write_csv(output_dir / "metrics" / "keywords_ranked.csv")

    pl.DataFrame(doc_rows).write_csv(output_dir / "raw" / "doc_info.csv")

    # ---- Aggregate per metric per K per algorithm ----
    metric_cols = []
    for k in ks:
        metric_cols += [f"P@{k}", f"R@{k}", f"F1@{k}", f"nDCG@{k}"]
    metric_cols += ["AP", "MRR"]

    summary_rows = []
    for name in ALGORITHMS:
        sub = df.filter(pl.col("algorithm") == name)
        for m in metric_cols:
            vals = sub[m].to_numpy()
            desc = stats_module.describe(vals)
            summary_rows.append({"algorithm": name, "metric": m, **desc})

    summary = pl.DataFrame(summary_rows)
    summary.write_csv(output_dir / "processed" / "summary.csv")

    # ---- Improvement BB-IDF vs TF-IDF (per doc, per K, per metric) ----
    improv_rows = []
    tf = df.filter(pl.col("algorithm") == "tfidf").sort("doc_id")
    bb = df.filter(pl.col("algorithm") == "bbidf").sort("doc_id")
    for m in metric_cols:
        a = tf[m].to_numpy()
        b = bb[m].to_numpy()
        for d_idx in range(len(a)):
            base = a[d_idx]
            imp = ((b[d_idx] - base) / base) * 100.0 if base != 0 else None
            improv_rows.append({
                "doc_id": int(tf["doc_id"][d_idx]),
                "metric": m, "tfidf": a[d_idx], "bbidf": b[d_idx],
                "diff": b[d_idx] - a[d_idx], "improvement_pct": imp,
            })
    improv_df = pl.DataFrame(improv_rows)
    improv_df.write_csv(output_dir / "processed" / "improvement_bbidf_vs_tfidf.csv")

    # ---- Statistical comparisons (paired, per metric per K) ----
    stat_rows = []
    for m in metric_cols:
        for name in ALGORITHMS:
            if name == "bbidf":
                continue
            a = df.filter(pl.col("algorithm") == name)[m].to_numpy()
            b = df.filter(pl.col("algorithm") == "bbidf")[m].to_numpy()
            order = np.argsort(df.filter(pl.col("algorithm") == name)["doc_id"])
            a = a[order]
            b = b[order]
            w = stats_module.paired_wilcoxon(a, b)
            ci = stats_module.bootstrap_mean_diff_ci(a, b, seed=seed)
            stat_rows.append({
                "metric": m, "comparison": f"bbidf_vs_{name}",
                "p_value": w["p_value"], "n_pairs": w["n_pairs"],
                "cohens_d": stats_module.cohens_d_paired(a, b),
                "rank_biserial": stats_module.rank_biserial(a, b),
                "mean_diff": ci["mean_diff"],
                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            })
    stat_df = pl.DataFrame(stat_rows)
    stat_df.write_csv(output_dir / "statistical" / "paired_tests.csv")

    # ---- TextRank gap ----
    gap_rows = []
    for m in metric_cols:
        tr = df.filter(pl.col("algorithm") == "textrank")[m].to_numpy()
        bb = df.filter(pl.col("algorithm") == "bbidf")[m].to_numpy()
        tfv = df.filter(pl.col("algorithm") == "tfidf")[m].to_numpy()
        tr_doc_ids = df.filter(pl.col("algorithm") == "textrank")["doc_id"].to_list()
        o = np.argsort(tr_doc_ids)
        tr, bb, tfv = tr[o], bb[o], tfv[o]
        for d_idx in range(len(tr)):
            gap_rows.append({
                "doc_id": tr_doc_ids[o[d_idx]],
                "metric": m, "textrank": tr[d_idx], "bbidf": bb[d_idx],
                "tfidf": tfv[d_idx],
                "bbidf_gap_to_textrank": bb[d_idx] - tr[d_idx],
                "bbidf_pct_of_textrank": (bb[d_idx] / tr[d_idx]) * 100.0 if tr[d_idx] != 0 else None,
            })
    pl.DataFrame(gap_rows).write_csv(
        output_dir / "processed" / "gap_to_textrank.csv")

    metadata = {
        "n_docs": n_docs,
        "n_gold_docs": n_gold_docs,
        "excluded_docs": excluded,
        "vocab_size": len(vocab),
        "ks": ks,
        "textrank_window": textrank_window,
        "algorithms": ALGORITHMS,
        "preprocessing": "spaCy es_core_news_sm: lemma.lower(); filter punct/space/stop/like_num/len>=3",
        "idf_formula": "ln((1+N)/(1+df)) + 1",
        "bbidf_band": "mu + 0.5*sigma / mu + 2.5*sigma over nonzero freqs; fallback [1.5,4.5] if band_inf>=band_sup or n_tokens<30",
        "bbidf_change_vs_tfidf": "df -> df_banda only (no hard zeroing of weights)",
        "seed": seed,
        "prep_time_s": round(t_prep, 4),
        "fit_time_s": {k: round(v, 4) for k, v in times.items()},
        "per_doc_time_s": {k: round(v / n_docs, 6) for k, v in times.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Top-K keyword lists for case analysis (K=10) ----
    top_rows = []
    for d in range(n_docs):
        if d in excluded:
            continue
        for name in ALGORITHMS:
            ranked = metrics_module.ranked_terms(weights[name][d], vocab, k=10)
            top_rows.append({
                "doc_id": d, "file": documents[d][0], "algorithm": name,
                "keywords": ", ".join(ranked),
            })
    pl.DataFrame(top_rows).write_csv(output_dir / "metrics" / "top10_keywords.csv")

    return {
        "metadata": metadata,
        "df": df,
        "summary": summary,
        "improv_df": improv_df,
        "stat_df": stat_df,
        "weights": weights,
        "vocab": vocab,
        "docs_tokens": docs_tokens,
        "gold_sets": gold_sets,
        "excluded": excluded,
        "documents": documents,
    }


def main():
    ap = argparse.ArgumentParser(description="BB-IDF keyword extraction evaluation")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--window", type=int, default=2,
                    help="TextRank co-occurrence window")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_experiment(corpus_dir=args.corpus, output_dir=args.out,
                         ks=args.ks, textrank_window=args.window, seed=args.seed)

    print("\n=== RESUMEN (media F1@K) ===")
    summ = res["summary"]
    for name in ALGORITHMS:
        vals = []
        for k in args.ks:
            f = summ.filter((pl.col("algorithm") == name) & (pl.col("metric") == f"F1@{k}"))
            vals.append(f"F1@{k}={f['mean'][0]:.3f}")
        print(f"  {ALGO_LABELS[name]:>8}: " + "  ".join(vals))

    print("\n=== Mejora media BB-IDF vs TF-IDF (F1@K) ===")
    imp = res["improv_df"]
    for k in args.ks:
        sub = imp.filter(pl.col("metric") == f"F1@{k}")
        print(f"  F1@{k}: {sub['improvement_pct'].mean():+.2f}%")

    print(f"\nResultados guardados en {args.out}/")


if __name__ == "__main__":
    main()
