"""Robustness and case analysis.

Produces:
  * results/statistical/robustness.csv   (TextRank window sensitivity + hard-band variant)
  * results/statistical/band_diagnostic.csv
  * results/tables/case_analysis.md
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

from bb_idf.experiment import gold as gold_module
from bb_idf.experiment import metrics as metrics_module
from bb_idf.experiment import pipeline as pipeline_module
from bb_idf.experiment import scorers as scorers_module

ALGO_LABELS = {"tfidf": "TF-IDF", "bbidf": "BB-IDF", "textrank": "TextRank"}


def _load(corpus_dir):
    prep = pipeline_module.preprocess_corpus(corpus_dir)
    return (prep.documents, prep.docs_tokens, prep.vocab, prep.X,
            prep.gold_sets, prep.excluded)


def _mean_metrics(W, vocab, gold_sets, excluded, ks=(5, 10, 20, 50)):
    vals = {f"F1@{k}": [] for k in ks}
    vals["AP"] = []
    vals["MRR"] = []
    for d in range(W.shape[0]):
        if d in excluded:
            continue
        ev = metrics_module.evaluate_document(W[d], vocab, gold_sets[d], list(ks))
        for k in ks:
            vals[f"F1@{k}"].append(ev[f"F1@{k}"])
        vals["AP"].append(ev["AP"])
        vals["MRR"].append(ev["MRR"])
    return {k: float(np.mean(v)) for k, v in vals.items()}


def run_robustness(resdir="results", corpus_dir="data/corpus"):
    resdir = Path(resdir)
    documents, docs_tokens, vocab, X, gold_sets, excluded = _load(corpus_dir)

    rows = []

    # TextRank window sensitivity
    for window in [2, 5, 10]:
        t = time.perf_counter()
        W = scorers_module.textrank_weights(docs_tokens, vocab, window=window)
        dt = time.perf_counter() - t
        m = _mean_metrics(W, vocab, gold_sets, excluded)
        rows.append({"variant": f"textrank_window={window}",
                     "F1@5": m["F1@5"], "F1@10": m["F1@10"],
                     "F1@20": m["F1@20"], "F1@50": m["F1@50"],
                     "AP": m["AP"], "MRR": m["MRR"], "time_s": round(dt, 4)})

    # Hard-band BB-IDF variant
    W = scorers_module.bbidf_weights_hard(X)
    m = _mean_metrics(W, vocab, gold_sets, excluded)
    rows.append({"variant": "bbidf_hard_band", "F1@5": m["F1@5"],
                 "F1@10": m["F1@10"], "F1@20": m["F1@20"], "F1@50": m["F1@50"],
                 "AP": m["AP"], "MRR": m["MRR"], "time_s": np.nan})

    # Baselines for reference
    for name, W in [("tfidf", scorers_module.tfidf_weights(X)),
                    ("bbidf", scorers_module.bbidf_weights(X)),
                    ("textrank", scorers_module.textrank_weights(docs_tokens, vocab))]:
        m = _mean_metrics(W, vocab, gold_sets, excluded)
        rows.append({"variant": name, "F1@5": m["F1@5"], "F1@10": m["F1@10"],
                     "F1@20": m["F1@20"], "F1@50": m["F1@50"],
                     "AP": m["AP"], "MRR": m["MRR"], "time_s": np.nan})

    df = pl.DataFrame(rows)
    (resdir / "statistical").mkdir(parents=True, exist_ok=True)
    df.write_csv(resdir / "statistical" / "robustness.csv")

    # Band diagnostic
    df_classic = scorers_module._document_frequency(X)
    df_banda = scorers_module._band_document_frequency(X)
    diag = pl.DataFrame({
        "term_count": [len(vocab)],
        "n_terms_df_banda_lt_df": [int((df_banda < df_classic).sum())],
        "n_terms_df_banda_zero": [int((df_banda == 0).sum())],
        "df_classic_mean": [float(df_classic.mean())],
        "df_banda_mean": [float(df_banda.mean())],
        "idf_corr": [float(np.corrcoef(
            scorers_module._idf_formula(X.shape[0], df_classic),
            scorers_module._idf_formula(X.shape[0], df_banda))[0, 1])],
    })
    diag.write_csv(resdir / "statistical" / "band_diagnostic.csv")

    return df, documents, docs_tokens, vocab, X, gold_sets, excluded


def run_cases(resdir="results", corpus_dir="data/corpus", k=10):
    """Identify and document best/worst cases for BB-IDF vs TF-IDF / TextRank."""
    resdir = Path(resdir)
    documents, docs_tokens, vocab, X, gold_sets, excluded = _load(corpus_dir)

    weights = {
        "tfidf": scorers_module.tfidf_weights(X),
        "bbidf": scorers_module.bbidf_weights(X),
        "textrank": scorers_module.textrank_weights(docs_tokens, vocab),
    }
    gold_by_file = gold_module.gold_by_file()

    f1 = {}
    for algo in weights:
        f1[algo] = {}
        for d in range(X.shape[0]):
            if d in excluded:
                continue
            ev = metrics_module.evaluate_document(weights[algo][d], vocab,
                                                  gold_sets[d], [k])
            f1[algo][d] = ev[f"F1@{k}"]

    docs_eval = sorted(f1["tfidf"].keys())
    adv = sorted(docs_eval, key=lambda d: -(f1["bbidf"][d] - f1["tfidf"][d]))
    worst = sorted(docs_eval, key=lambda d: (f1["bbidf"][d] - f1["tfidf"][d]))
    close_tr = sorted(docs_eval, key=lambda d: abs(f1["bbidf"][d] - f1["textrank"][d]))
    beat_tr = [d for d in docs_eval if f1["bbidf"][d] > f1["textrank"][d]]

    def _fmt(d, algo):
        ranked = metrics_module.ranked_terms(weights[algo][d], vocab, k=10)
        return ", ".join(ranked)

    lines = ["# Análisis de casos (F1@10)", ""]
    lines.append(f"n evaluados: {len(docs_eval)} documentos (excluidos: {excluded})")
    lines.append("")

    def _case(title, doc_ids):
        lines.append(f"## {title}")
        lines.append("")
        for d in doc_ids:
            fname = documents[d][0]
            gset = gold_sets[d]
            gold_norm = ", ".join(sorted(gset))
            lines.append(f"### Doc {d}: {fname}")
            lines.append(f"- Gold (autor): {gold_norm}")
            lines.append(f"- F1@10: TF-IDF={f1['tfidf'][d]:.3f} | "
                         f"BB-IDF={f1['bbidf'][d]:.3f} | "
                         f"TextRank={f1['textrank'][d]:.3f}")
            lines.append("- Top-10 TF-IDF: " + _fmt(d, "tfidf"))
            lines.append("- Top-10 BB-IDF: " + _fmt(d, "bbidf"))
            lines.append("- Top-10 TextRank: " + _fmt(d, "textrank"))
            lines.append("")

    _case("Mayor ventaja de BB-IDF sobre TF-IDF", adv[:5])
    _case("Peor desempeño relativo de BB-IDF", worst[:5])
    _case("BB-IDF más cercano a TextRank", close_tr[:5])
    _case("BB-IDF supera a TextRank", beat_tr[:10] if beat_tr else [])

    (resdir / "tables").mkdir(parents=True, exist_ok=True)
    (resdir / "tables" / "case_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"  Casos guardados en {resdir / 'tables' / 'case_analysis.md'}")


if __name__ == "__main__":
    run_robustness()
    run_cases()
