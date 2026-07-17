from pathlib import Path

from tf_inst.algorithms import TfidfVectorizerWrapper, TextRankVectorizer, TFPDC_Scalable
from tf_inst.evaluation.benchmark import Benchmark
from tf_inst.utils.io import load_text_files, save_results


def main():
    corpus_dir = Path("data/corpus")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    docs_raw = load_text_files(corpus_dir)
    documents = [content for _, content in docs_raw]
    queries = documents[:5]

    benchmark = Benchmark()
    vectorizers = {
        "tfidf": TfidfVectorizerWrapper(),
        "textrank": TextRankVectorizer(),
        "tfpdc": TFPDC_Scalable(),
    }

    df = benchmark.run(documents, queries, vectorizers)
    save_results(df, results_dir / "benchmark.csv")
    print(df)


if __name__ == "__main__":
    main()
