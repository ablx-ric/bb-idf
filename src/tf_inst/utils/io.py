from pathlib import Path
import polars as pl


def load_text_files(corpus_dir: str | Path) -> list[tuple[str, str]]:
    """Carga documentos .txt de un directorio.

    Returns:
        Lista de (nombre_archivo, contenido).
    """
    path = Path(corpus_dir)
    docs = []
    for f in sorted(path.glob("*.txt")):
        docs.append((f.stem, f.read_text(encoding="utf-8")))
    return docs


def save_results(df: pl.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
