from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

from bb_idf.evaluation.benchmark import AlgoMetrics

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']


def _apply_style():
    mpl.rcParams.update({
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.5,
        'axes.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.prop_cycle': mpl.cycler(color=OKABE_ITO),
        'xtick.major.size': 3,
        'xtick.labelsize': 7,
        'ytick.major.size': 3,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'image.cmap': 'viridis',
    })


def _save(fig: plt.Figure, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{name}.png", dpi=300, bbox_inches='tight')
    print(f"  OK {name}.png")
    plt.close(fig)


def _bar_plot(ax, names: list[str], values: list[float], ylabel: str,
              fmt_val: str = '{:.2f}', colors=None):
    x = np.arange(len(names))
    width = 0.55
    cs = (colors or OKABE_ITO)[:len(names)]
    bars = ax.bar(x, values, width, color=cs, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt_val.format(v), ha='center', va='bottom', fontsize=7)


def plot_benchmark_bars(metrics: list[AlgoMetrics], output_dir: Path):
    _apply_style()
    names = [m.name for m in metrics]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _bar_plot(ax, names, [m.fit_time for m in metrics],
              'Tiempo de ajuste (s)', '{:.2f}s')
    _save(fig, output_dir, 'benchmark_fit_time')

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _bar_plot(ax, names, [m.vocab_size for m in metrics],
              'Tamanio del vocabulario', '{:d}')
    _save(fig, output_dir, 'benchmark_vocab_size')

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _bar_plot(ax, names, [m.transform_time for m in metrics],
              'Tiempo de transformacion (s)', '{:.4f}s')
    _save(fig, output_dir, 'benchmark_transform_time')

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _bar_plot(ax, names, [m.query_time for m in metrics],
              'Tiempo de consulta (s)', '{:.4f}s')
    _save(fig, output_dir, 'benchmark_query_time')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    _bar_plot(ax1, names, [m.fit_time for m in metrics], 'Tiempo de ajuste (s)', '{:.2f}s')
    _bar_plot(ax2, names, [m.vocab_size for m in metrics], 'Tamanio del vocabulario', '{:d}')
    _save(fig, output_dir, 'benchmark_panel_times')


def plot_sparsity_density(metrics: list[AlgoMetrics], output_dir: Path):
    _apply_style()
    names = [m.name for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    _bar_plot(ax1, names, [m.sparsity for m in metrics], 'Sparsity', '{:.4f}')
    _bar_plot(ax2, names, [m.density for m in metrics], 'Density', '{:.6f}')
    _save(fig, output_dir, 'benchmark_sparsity_density')


def plot_memory(metrics: list[AlgoMetrics], output_dir: Path):
    _apply_style()
    names = [m.name for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    _bar_plot(ax1, names, [m.matrix_memory_kb for m in metrics],
              'Memoria de matriz (KB)', '{:.1f}')
    _bar_plot(ax2, names, [m.serialized_size_kb for m in metrics],
              'Tamanio serializado (KB)', '{:.1f}')
    _save(fig, output_dir, 'benchmark_memory')


def plot_retrieval_metrics(metrics: list[AlgoMetrics], output_dir: Path):
    _apply_style()
    names = [m.name for m in metrics]
    ks = [1, 3, 5, 10]
    x = np.arange(len(ks))
    width = 0.25

    for metric_name, key_fn in [('precision', lambda m: m.precision_scores),
                                 ('recall', lambda m: m.recall_scores),
                                 ('ndcg', lambda m: m.ndcg_scores)]:
        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        for i, m in enumerate(metrics):
            vals = [key_fn(m)[k] for k in ks]
            offset = (i - (len(names) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=m.name,
                          color=OKABE_ITO[i], edgecolor='black', linewidth=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in ks])
        ax.set_ylabel(metric_name.capitalize())
        ax.legend(fontsize=7)
        _save(fig, output_dir, f'{metric_name}_at_k')

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _bar_plot(ax, names, [m.map_score for m in metrics], 'MAP', '{:.4f}')
    _save(fig, output_dir, 'mean_average_precision')

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _bar_plot(ax, names, [m.mrr_score for m in metrics], 'MRR', '{:.4f}')
    _save(fig, output_dir, 'mean_reciprocal_rank')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    _bar_plot(ax1, names, [m.map_score for m in metrics], 'MAP', '{:.4f}')
    _bar_plot(ax2, names, [m.mrr_score for m in metrics], 'MRR', '{:.4f}')
    _save(fig, output_dir, 'ranking_panel')


def plot_similarity_heatmaps(metrics: list[AlgoMetrics], output_dir: Path):
    _apply_style()
    for m in metrics:
        mat = m.sim_matrix
        if mat.size == 0:
            continue
        n_queries, n_docs = mat.shape
        figsize = (max(3, n_docs * 0.3), max(2.5, n_queries * 0.3))
        fig, ax = plt.subplots(figsize=(min(figsize[0], 8), min(figsize[1], 6)))
        sns.heatmap(mat, annot=n_queries <= 20 and n_docs <= 20,
                    fmt='.2f', cmap='viridis',
                    xticklabels=min(n_docs, 30), yticklabels=min(n_queries, 30),
                    cbar_kws={'label': 'Similitud coseno'}, ax=ax)
        ax.set_xlabel('Documentos')
        ax.set_ylabel('Consultas')
        ax.set_title(f'{m.name} - Matriz de similitud')
        _save(fig, output_dir, f'similarity_{m.name}')


def plot_weight_distribution(metrics: list[AlgoMetrics], output_dir: Path):
    _apply_style()
    for m in metrics:
        values = m.weight_matrix[m.weight_matrix > 0].flatten()
        if len(values) == 0:
            print(f"  ADVERTENCIA: {m.name} no tiene pesos positivos para graficar")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
        ax1.hist(values, bins=50, color=OKABE_ITO[0], edgecolor='black', linewidth=0.3)
        ax1.set_xlabel('Peso')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title(f'{m.name} - Histograma')

        ax2.boxplot(values, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=OKABE_ITO[1], alpha=0.6),
                    medianprops=dict(color='black'))
        ax2.set_ylabel('Peso')
        ax2.set_xticklabels([m.name])
        ax2.set_title('Boxplot')
        _save(fig, output_dir, f'weight_distribution_{m.name}')


def plot_all(metrics: list[AlgoMetrics], output_dir: Path):
    figures_dir = output_dir / 'figures'
    print("\nGenerando graficos...")
    plot_benchmark_bars(metrics, figures_dir)
    plot_sparsity_density(metrics, figures_dir)
    plot_memory(metrics, figures_dir)
    plot_retrieval_metrics(metrics, figures_dir)
    plot_similarity_heatmaps(metrics, figures_dir)
    plot_weight_distribution(metrics, figures_dir)
    print(f"  Graficos guardados en: {figures_dir}")
