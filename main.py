from pathlib import Path

from tqdm import tqdm

from tf_inst.algorithms import TfidfVectorizerWrapper, TextRankVectorizer, TFPDC_Scalable
from tf_inst.evaluation.benchmark import Benchmark
from tf_inst.utils.io import load_text_files, save_results


def main():
    corpus_dir = Path("data/corpus")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando documentos desde {corpus_dir} ...")
    docs_raw = load_text_files(corpus_dir)
    documents = [content for _, content in tqdm(docs_raw, desc="Procesando")]
    queries = documents[:5]

    print(f"Documentos cargados: {len(documents)}")
    print(f"Consultas (primeros {len(queries)} docs)")

    benchmark = Benchmark()
    vectorizers = {
        "tfidf": TfidfVectorizerWrapper(),
        "textrank": TextRankVectorizer(),
        "tfpdc": TFPDC_Scalable(),
    }

    df = benchmark.run(documents, queries, vectorizers)
    save_results(df, results_dir / "benchmark.csv")
    print("\nResultados:")
    for row in df.iter_rows(named=True):
        print(f"  {row['algorithm']:>8}  |  vocab={row['vocab_size']:<5}  |  shape={row['matrix_shape']:<12}  |  fit_time={row['fit_time_s']}s")


if __name__ == "__main__":
    main()
