"""LEGACY / OBSOLETO — evaluación de recuperación auto-referencial.

Usado únicamente por ``main.py``. No evalúa calidad de extracción de keywords;
la evaluación vigente está en ``bb_idf.experiment`` (orquestada por
``run_all.py``). Se conserva solo por referencia histórica.
"""

from .metrics import Evaluator
from .benchmark import Benchmark

__all__ = ["Evaluator", "Benchmark"]
