"""Genera todos los resultados del experimento con un solo comando.

Uso:
    uv run python run_all.py

Ejecuta en orden:
    0. pytest (opcional, --skip-tests para omitir)
    1. Experimento  (bb_idf.experiment.run)         -> results/{raw,processed,metrics,statistical}, metadata.json
    2. Figuras      (bb_idf.experiment.plots)       -> results/figures/
    3. Robustez + casos (bb_idf.experiment.robustness) -> results/statistical/, results/tables/
    4. Similitud vs TextRank (bb_idf.experiment.similarity) -> results/processed/, results/figures/
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print(f"\n==> {' '.join(cmd)}\n")
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Genera todos los resultados de bb-idf")
    ap.add_argument("--skip-tests", action="store_true",
                    help="Omitir la ejecución de pytest")
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50])
    ap.add_argument("--window", type=int, default=2,
                    help="Ventana de co-ocurrencia de TextRank")
    args = ap.parse_args()

    py = sys.executable

    if not args.skip_tests:
        print("=" * 60)
        print("PASO 0: pruebas")
        print("=" * 60)
        if _run([py, "-m", "pytest", "-q"]) != 0:
            print("Las pruebas fallaron. Corrige antes de continuar.")
            return 1

    print("=" * 60)
    print("PASO 1/4: experimento (metricas + estadistica + eficiencia)")
    print("=" * 60)
    cmd1 = [py, "-m", "bb_idf.experiment.run", "--ks", *map(str, args.ks),
            "--window", str(args.window)]
    if _run(cmd1) != 0:
        print("El experimento falló.")
        return 1

    print("=" * 60)
    print("PASO 2/4: figuras")
    print("=" * 60)
    if _run([py, "-m", "bb_idf.experiment.plots"]) != 0:
        print("La generación de figuras falló.")
        return 1

    print("=" * 60)
    print("PASO 3/4: robustez + analisis de casos")
    print("=" * 60)
    if _run([py, "-m", "bb_idf.experiment.robustness"]) != 0:
        print("La robustez/casos falló.")
        return 1

    print("=" * 60)
    print("PASO 4/4: similitud vs TextRank (suplementario)")
    print("=" * 60)
    if _run([py, "-m", "bb_idf.experiment.similarity"]) != 0:
        print("La similitud falló.")
        return 1

    print("\n" + "=" * 60)
    print("Listo. Resultados en results/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
