import sys
from pathlib import Path

import polars as pl
import fitz


def load_text_files(corpus_dir: str | Path) -> list[tuple[str, str]]:
    path = Path(corpus_dir)
    docs: list[tuple[str, str]] = []

    for f in sorted(path.glob("*.txt")):
        try:
            docs.append((f.stem, f.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            print(f"  ADVERTENCIA: {f.name} no es UTF-8, omitiendo", file=sys.stderr)

    for f in sorted(path.glob("*.pdf")):
        try:
            doc = fitz.open(f)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            docs.append((f.stem, text))
        except Exception as exc:
            print(f"  ADVERTENCIA: {f.name} no se pudo leer ({exc}), omitiendo",
                  file=sys.stderr)

    return docs


def save_results(df: pl.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
