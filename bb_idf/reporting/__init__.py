"""LEGACY / OBSOLETO — gráficos y estadística del benchmark computacional.

Usado únicamente por ``main.py``. La generación de figuras y el análisis
estadístico vigentes están en ``bb_idf.experiment.plots`` y
``bb_idf.experiment.stats``. Se conserva solo por referencia histórica.
"""

from .graphs import plot_all
from .stats import compare_algorithms, format_report

__all__ = ["plot_all", "compare_algorithms", "format_report"]
