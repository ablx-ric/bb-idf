import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm

from bb_idf.algorithms import TfidfVectorizerWrapper, TextRankVectorizer, BBIDF
from bb_idf.evaluation.benchmark import Benchmark
from bb_idf.utils.io import load_text_files, save_results
from bb_idf.reporting import plot_all, compare_algorithms, format_report


def _build_vectorizers():
    return {
        "tfidf": TfidfVectorizerWrapper(),
        "textrank": TextRankVectorizer(),
        "bbidf": BBIDF(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="bb-idf: comparacion de algoritmos de ponderacion de terminos"
    )
    parser.add_argument("--graphs", action="store_true",
                        help="Generar graficos de los resultados")
    parser.add_argument("--runs", type=int, default=1,
                        help="Numero de ejecuciones para media +/- std (defecto: 1)")
    parser.add_argument("--scalability", action="store_true",
                        help="Prueba de escalabilidad con subconjuntos del corpus")
    args = parser.parse_args()

    corpus_dir = Path("data/corpus")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando documentos desde {corpus_dir} ...")
    docs_raw = load_text_files(corpus_dir)
    documents = [content for _, content in tqdm(docs_raw, desc="Procesando")]
    queries = documents[:5]

    print(f"Documentos cargados: {len(documents)}")
    print(f"Consultas (primeros {len(queries)} docs)")

    if args.runs > 1:
        print(f"\nEjecutando benchmark ({args.runs} corridas)...")
        all_runs = []
        last_benchmark = None
        for run_idx in range(args.runs):
            benchmark = Benchmark()
            vectorizers = _build_vectorizers()
            df = benchmark.run(documents, queries, vectorizers)
            df = df.with_columns(pl.lit(run_idx + 1).alias("run"))
            all_runs.append(df)
            last_benchmark = benchmark

        df_all = pl.concat(all_runs)
        df_summary = df_all.group_by("algorithm").agg([
            pl.col("fit_time_s").mean().round(4).alias("fit_time_mean"),
            pl.col("fit_time_s").std().round(4).alias("fit_time_std"),
            pl.col("transform_time_s").mean().round(4).alias("transform_time_mean"),
            pl.col("transform_time_s").std().round(4).alias("transform_time_std"),
            pl.col("query_time_s").mean().round(4).alias("query_time_mean"),
            pl.col("query_time_s").std().round(4).alias("query_time_std"),
            pl.col("MAP").mean().round(4).alias("MAP_mean"),
            pl.col("MAP").std().round(4).alias("MAP_std"),
            pl.col("MRR").mean().round(4).alias("MRR_mean"),
            pl.col("MRR").std().round(4).alias("MRR_std"),
            pl.col("sparsity").mean().round(6).alias("sparsity_mean"),
            pl.col("matrix_memory_kb").mean().round(2).alias("memory_kb_mean"),
        ]).sort("algorithm")

        save_results(df_all, output_dir / "benchmark" / "benchmark_all_runs.csv")
        save_results(df_summary, output_dir / "benchmark" / "benchmark_summary.csv")

        print("\nResumen (media +/- std):")
        for row in df_summary.iter_rows(named=True):
            print(f"  {row['algorithm']:>8}  |  "
                  f"fit={row['fit_time_mean']}s +/-{row['fit_time_std']}s  |  "
                  f"MAP={row['MAP_mean']} +/-{row['MAP_std']}  |  "
                  f"sparsity={row['sparsity_mean']}")

        fit_data = {
            row['algorithm']: df_all.filter(
                pl.col('algorithm') == row['algorithm']
            )['fit_time_s'].to_list()
            for row in df_summary.iter_rows(named=True)
        }
        if len(fit_data) >= 2:
            stat_result = compare_algorithms(fit_data)
            report = format_report(stat_result)
            print(f"\n{report}")
            stats_path = output_dir / "metrics" / "statistical_analysis.txt"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(report, encoding="utf-8")
            print(f"  Analisis estadistico guardado: {stats_path}")

        if args.graphs and last_benchmark is not None:
            plot_all(last_benchmark.result.metrics, output_dir)
    else:
        benchmark = Benchmark()
        vectorizers = _build_vectorizers()
        df = benchmark.run(documents, queries, vectorizers)
        save_results(df, output_dir / "benchmark" / "benchmark.csv")

        print("\nResultados:")
        headers = ["algoritmo", "vocab", "shape", "fit(s)", "transf(s)", "query(s)",
                   "sparsity", "mem(KB)", "P@5", "R@5", "nDCG@5", "MAP", "MRR"]
        print(f"  {' | '.join(f'{h:>10}' for h in headers)}")
        print("  " + "-" * (13 * len(headers)))
        for row in df.iter_rows(named=True):
            vals = [row['algorithm'], row['vocab_size'], row['matrix_shape'],
                    f"{row['fit_time_s']:.3f}", f"{row['transform_time_s']:.4f}",
                    f"{row['query_time_s']:.4f}", f"{row['sparsity']:.4f}",
                    row['matrix_memory_kb'], f"{row['precision@5']:.3f}",
                    f"{row['recall@5']:.3f}", f"{row['ndcg@5']:.3f}",
                    f"{row['MAP']:.4f}", f"{row['MRR']:.4f}"]
            print(f"  {' | '.join(f'{str(v):>10}' for v in vals)}")

        for res in benchmark.result.metrics:
            metrics_path = output_dir / "metrics" / f"{res.name}_metrics.csv"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            sim_df = pl.DataFrame(res.sim_matrix)
            sim_df.write_csv(metrics_path)

        print(f"  Benchmark guardado: {output_dir / 'benchmark' / 'benchmark.csv'}")

        if args.graphs:
            plot_all(benchmark.result.metrics, output_dir)

    if args.scalability:
        print("\nEjecutando prueba de escalabilidad...")
        sizes_path = output_dir / "metrics" / "scalability.csv"
        sizes_path.parent.mkdir(parents=True, exist_ok=True)
        scalability_rows = []
        sizes = [5, 10, 20]
        if len(documents) not in sizes:
            sizes.append(len(documents))
        for size in sizes:
            subset = documents[:size]
            q_sub = queries[:min(5, size)]
            print(f"  Probando con {len(subset)} documentos...")
            b = Benchmark()
            v = _build_vectorizers()
            df_s = b.run(subset, q_sub, v)
            for row in df_s.iter_rows(named=True):
                scalability_rows.append({
                    "n_docs": len(subset),
                    "algorithm": row['algorithm'],
                    "fit_time_s": row['fit_time_s'],
                    "transform_time_s": row['transform_time_s'],
                    "query_time_s": row['query_time_s'],
                    "matrix_memory_kb": row['matrix_memory_kb'],
                    "sparsity": row['sparsity'],
                })
        df_scal = pl.DataFrame(scalability_rows)
        df_scal.write_csv(sizes_path)
        print(f"  Escalabilidad guardada: {sizes_path}")


if __name__ == "__main__":
    main()
