# bb-idf

Evaluación científica de algoritmos de extracción de keywords sobre un corpus
de documentos académicos de turismo. Compara **TF-IDF**, **BB-IDF** (propuesta)
y **TextRank** (referencia/benchmark), tanto en **calidad** (contra las palabras
clave declaradas por los autores) como en **costo computacional**.

## Instalación

```bash
uv sync
python -m spacy download es_core_news_sm
```

## Uso

Hay dos vías de ejecución:

### 1. Evaluación de extracción de keywords (principal)

```bash
# Métricas por documento/algoritmo/K + estadística + eficiencia
uv run python -m bb_idf.experiment.run

# Figuras científicas (results/figures/)
uv run python -m bb_idf.experiment.plots

# Robustez (ventana de TextRank, variante de banda dura, diagnóstico) + casos
uv run python -m bb_idf.experiment.robustness
```

Estructura de salida:

```
results/
├── raw/            per_doc_metrics.csv, doc_info.csv
├── processed/      summary.csv, improvement_bbidf_vs_tfidf.csv, gap_to_textrank.csv
├── metrics/        top10_keywords.csv, keywords_ranked.csv
├── statistical/    paired_tests.csv, robustness.csv, band_diagnostic.csv
├── figures/        9 figuras PNG
├── tables/         case_analysis.md
├── metadata.json   configuración reproducible
└── INFORME_EXPERIMENTAL.md
```

### 2. Benchmark legado (solo eficiencia)

```bash
uv run python main.py              # corrida única
uv run python main.py --runs 10    # media +/- std (tiempos)
uv run python main.py --graphs     # 15 gráficos en output/figures/
uv run python main.py --scalability
```

> **Nota metodológica**: el benchmark de `main.py` usa `consulta = documento`
> (cada consulta es idéntica a un documento y solo ese documento es relevante),
> por lo que sus métricas de *recuperación* (P@k, MAP, MRR, nDCG) son
> degeneradas (valen 1.0 para los tres algoritmos). Úsese solo para comparar
> **eficiencia** (tiempos, memoria, dispersión). La evaluación de **calidad**
> es la del experimento de keywords (§1).

## Corpus

- **33 documentos PDF** (tesis y artículos de turismo; región Amazonas/Chachapoyas, Perú).
  *(El README y `data/qrels/queries.json` decían 34: falta un PDF.)*
- 31/33 documentos declaran **"Palabras clave(s)" / "Keywords"** del autor →
  se usan como **gold standard** (no inventado).
- 30 documentos forman el conjunto de evaluación (3 excluidos sin keywords
  utilizables). Ver `bb_idf/experiment/gold.py`.

## Preprocesamiento (idéntico para los 3 algoritmos y el gold)

1. Extracción de texto: PyMuPDF (`fitz`).
2. Tokenización + lematización: spaCy `es_core_news_sm` (sin parser/ner), `lemma_.lower()`.
3. Filtros: sin puntuación/espacios, sin stopwords (ES + mínima lista EN por el
   contenido bilingüe), sin tokens numéricos (`like_num`), **lema alfabético**
   (`isalpha`, elimina ISSN/DOI/URLs), longitud ≥ 3.
4. **Vocabulario compartido** (unión de los 33 docs) y **matriz de conteos
   compartida**: los tres algoritmos operan sobre el mismo espacio de términos.

El gold se normaliza con el mismo pipeline y se aplana a unigramas de contenido.

## Algoritmos

### TF-IDF

$$w_{t,d} = tf_{t,d} \cdot idf(t), \qquad idf(t) = \ln\frac{1+N}{1+df(t)} + 1$$

### TextRank

Grafo de co-ocurrencia por documento + PageRank ponderado (Mihalcea & Tarau 2004):

$$S(v_i) = (1-d) + d \cdot \sum_{v_j \in In(v_i)} \frac{w_{ji}}{\sum_{v_k \in Out(v_j)} w_{jk}} S(v_j)$$

> **Corrección**: el código (`bb_idf/algorithms/textrank.py`) usa ventana
> `window_size=2` por defecto; el README anterior y el notebook decían 5. El
> experimento usa ventana 2 (mejor caso reportado en M&T 2004) y evalúa 2/5/10
> en robustez.

### BB-IDF (Bounded Band Inverse Document Frequency)

Evolución del TF-IDF que introduce un **filtro de banda estadístico adaptativo
por documento** en el conteo de la frecuencia documental.

- Banda: $[\,\mu_d + 0.5\sigma_d,\; \mu_d + 2.5\sigma_d\,]$, con $\mu_d, \sigma_d$
  sobre las frecuencias no nulas del documento `d` (σ poblacional).
- Fallback `[1.5, 4.5]` si `banda_inf ≥ banda_sup` o el doc tiene < 30 tokens.
- $df_{banda}(t) = |\{d : banda\_inf_d \le f_{t,d} \le banda\_sup_d\}|$.
- $w_{t,d} = tf_{t,d} \cdot idf_{banda}(t)$, con la **misma** fórmula IDF que
  TF-IDF ($idf_{banda} = \ln\frac{1+N}{1+df_{banda}(t)} + 1$).

**Variantes consideradas** (ver `bb_idf/experiment/scorers.py`):

| Variante | Definición | Resultado |
|---|---|---|
| `bbidf` (principal) | solo cambia `df → df_banda` | comparable a TF-IDF, iguala a TextRank |
| `bbidf_hard_band` | además anula pesos fuera de banda (como `bb_idf/algorithms/bbidf.py`) | **inviable** en docs largos (F1@10 ≈ 0.04) |

## Protocolo experimental

| Elemento | Valor |
|---|---|
| Unidad experimental | documento (n = 30 evaluables) |
| K | 5, 10, 20, 50 |
| Métricas | P@K, R@K, F1@K, AP, MRR, nDCG@K |
| Comparación principal | BB-IDF vs TF-IDF (pareada por documento) |
| Comparación secundaria | BB-IDF / TF-IDF vs TextRank |
| Estadística | Wilcoxon signed-rank, bootstrap CI 95%, Cohen's d (pareado), rank-biserial |

## Resultados principales

Media F1@K y ranking (30 documentos):

| Métrica | TF-IDF | BB-IDF | TextRank |
|---|---|---|---|
| F1@5 | 0.432 | 0.442 | **0.455** |
| F1@10 | 0.437 | **0.477** | 0.467 |
| F1@20 | 0.348 | 0.347 | 0.349 |
| F1@50 | 0.190 | 0.190 | 0.185 |
| AP | 0.488 | 0.500 | **0.508** |
| MRR | 0.741 | 0.747 | **0.828** |

Mejora de BB-IDF sobre TF-IDF (F1@K):

| K | Mejora media | p (Wilcoxon) | Cohen's d |
|---|---|---|---|
| 5 | +3.4% | 0.266 | +0.10 |
| **10** | **+10.3%** | **0.006** | **+0.56** |
| 20 | −0.9% | 0.750 | −0.04 |
| 50 | +0.5% | 0.625 | +0.02 |

Eficiencia (33 docs, sin preprocesado):

| Algoritmo | Tiempo total | Tiempo/doc | Factor vs TF-IDF |
|---|---|---|---|
| TF-IDF | 0.0042 s | 0.00013 s | 1× |
| BB-IDF | 0.0089 s | 0.00027 s | 2.1× |
| TextRank | 4.117 s | 0.125 s | ~980× |

## Hallazgos clave

1. **BB-IDF mejora a TF-IDF de forma modesta y acotada**: significativa solo en
   **F1@10 / R@10** (p = 0.006, d = 0.56). No significativa en K = 5, 20, 50 ni
   en AP/MRR. La mejora no es uniforme (gana en 10/30, empata en 19/30).
2. **BB-IDF está a la par de TextRank** (sin diferencias significativas; en
   F1@10 lo iguala o supera ligeramente).
3. **El filtro de banda con umbrales sobre conteos crudos colapsa el IDF**: el
   91% de los términos tienen `df_banda = 0` (correlación IDF clásico/banda =
   0.59), por lo que BB-IDF (solo-df) se comporta como un ranking por
   frecuencia con penalización selectiva.
4. **La variante con filtro duro** (la implementada en
   `bb_idf/algorithms/bbidf.py`) anula ~91% de términos y es inviable para
   extracción de keywords (F1@10 ≈ 0.04).
5. El benchmark legado (`main.py`) no mide calidad (ground truth degenerado);
   véase §2.

## Limitaciones (n = 30)

- Potencia limitada; conclusiones exploratorias.
- Gold de unigramas (los algoritmos son unigrama; no se evalúan frases completas).
- Lematizador español sobre el artículo en inglés y fragmentos bilingües.
- Muchos empates entre BB-IDF y TF-IDF (≈ 19/30 en F1@10).

## Referencias

- Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into Text. *EMNLP*.
- Ramos, J. (2003). Using TF-IDF to Determine Word Relevance in Document Queries. *ICML*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
