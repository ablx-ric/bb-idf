# tf-inst

Comparacion de algoritmos de ponderacion de terminos (TF-IDF, TextRank, BB-IDF) para recuperacion de informacion en espanol.

## Instalacion

```bash
uv sync
python -m spacy download es_core_news_sm
```

## Uso

```bash
# Benchmark simple (34 docs, corrida unica)
uv run python main.py

# Multi-run con media +/- desviacion y analisis estadistico
uv run python main.py --runs 10

# Generar graficos (15 graficos en output/figures/)
uv run python main.py --graphs

# Prueba de escalabilidad (subconjuntos de 5,10,20,34 docs)
uv run python main.py --scalability

# Combinar flags
uv run python main.py --runs 10 --graphs --scalability
```

### Estructura de output

```
output/
├── benchmark/
│   ├── benchmark.csv              # Resultados corrida unica
│   ├── benchmark_all_runs.csv      # Resultados multi-run (raw)
│   └── benchmark_summary.csv       # Media +/- std por algoritmo
├── figures/
│   ├── benchmark_fit_time.png      # Tiempo de ajuste
│   ├── ...                         # (15 graficos, ver seccion)
└── metrics/
    ├── scalability.csv             # Tiempos por tamano de corpus
    ├── statistical_analysis.txt    # ANOVA + post-hoc
    ├── tfidf_metrics.csv           # Matriz de similitud
    ├── textrank_metrics.csv
    └── bbidf_metrics.csv
```

## Algoritmos

### TF-IDF

Implementacion mediante `TfidfVectorizer` de scikit-learn. Ponderacion classica:

$$w_{t,d} = tf_{t,d} \cdot \log\frac{N}{df_t}$$

### TextRank

Implementacion propia basada en Mihalcea & Tarau (2004). Construye un grafo de co-ocurrencia de terminos (ventana deslizante de tamanio 5) y ejecuta PageRank iterativo hasta convergencia:

$$S(v_i) = (1-d) + d \cdot \sum_{v_j \in In(v_i)} \frac{w_{ji}}{\sum_{v_k \in Out(v_j)} w_{jk}} S(v_j)$$

### BB-IDF (Propuesta)

**Bounded Band Inverse Document Frequency** — evolucion del TF-IDF que introduce un **filtro de banda estadistico adaptativo por documento**. En lugar de contar un documento para el DF con una sola mencion del termino, BB-IDF exige que la frecuencia del termino caiga dentro de una banda estadistica especifica de ese documento.

#### Componentes

- **Banda inferior** ($\mu_d + 0.5\sigma_d$): Filtra **ruido** — terminos que aparecen muy pocas veces en el documento. Una mencion aislada no implica que el documento trate sobre ese tema.
- **Banda superior** ($\mu_d + 2.5\sigma_d$): Filtra **spam** (keyword stuffing) y stop words — terminos que aparecen excesivamente (ej. "el", "de" en documentos grandes).
- **Fallback para docs cortos** (< 30 tokens): Banda fija $[1.5, 4.5]$ para documentos pequenos donde la distribucion estadistica no es confiable.

#### Formulas

Para cada documento $d$ con frecuencias $f_{t,d}$ y $\mu_d, \sigma_d$ calculados sobre los terminos presentes en $d$:

$$banda_d = [\mu_d + 0.5\sigma_d,\; \mu_d + 2.5\sigma_d]$$

El conteo de documentos filtrado por banda:

$$df_{banda}(t) = |\{ d : f_{t,d} \in banda_d \}|$$

$$BB\text{-}IDF(t) = \log\left(1 + \frac{N}{df_{banda}(t) + 1}\right)$$

$$w_{t,d} = f_{t,d} \cdot BB\text{-}IDF(t)$$

## Metodologia de Benchmark

### Pipeline

```
PDFs (34 docs)
  → extraccion de texto (pymupdf)
  → preprocesamiento (spaCy: lematizacion + stopwords ES)
  → fit_transform (cada algoritmo genera matriz term-documento)
  → transform (5 consultas = primeros 5 docs)
  → similitud coseno entre consultas y documentos
  → ranking descendente por similitud
  → calculo de metricas de recuperacion
```

### Relevancia (ground truth)

Cada consulta $q_i$ tiene como unico documento relevante a $d_i$ (el mismo indice). Esto implica que las metricas de recuperacion (MAP, MRR, Precision@k, Recall@k, nDCG@k) miden que tan bien cada algoritmo **reconstruye la identidad del documento original**, no su capacidad de recuperacion general. Con esta configuracion, los 3 algoritmos producen metricas identicas (ver seccion correspondiente).

## Definicion de Metricas

### Tiempo de ajuste (`fit_time`)

Tiempo que tarda cada algoritmo en construir el vocabulario y la matriz term-documento a partir de los 34 documentos preprocesados. Fundamental para comparar eficiencia.

### Tiempo de transformacion (`transform_time`)

Tiempo que tarda en vectorizar las 5 consultas usando el vocabulario ya aprendido. Refleja el costo de inferencia.

### Tiempo de consulta (`query_time`)

Tiempo que tarda en calcular la matriz de similitud coseno $Q \cdot D^T$ (5 consultas $\times$ 34 documentos) y generar el ranking.

### Sparsity y Density

- **Sparsity**: $1 - \frac{nnz}{total\_cells}$. Fraccion de celdas con valor cero.
- **Density**: $\frac{nnz}{total\_cells}$. Fraccion de celdas con valor distinto de cero.

Matrices mas densas implican que cada termino aparece en mas documentos.

### Memoria de matriz (`matrix_memory_kb`)

$D.nbytes / 1024$. Tamanio en memoria de la matriz term-documento densa (numpy array).

### Precision@k

$$P@k = \frac{|relevant \cap retrieved_k|}{k}$$

Fraccion de documentos relevantes entre los primeros $k$ recuperados.

### Recall@k

$$R@k = \frac{|relevant \cap retrieved_k|}{|relevant|}$$

Fraccion de documentos relevantes recuperados entre los primeros $k$.

### nDCG@k

$$DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$
$$nDCG_k = \frac{DCG_k}{IDCG_k}$$

Mide la calidad del ranking ponderando por posicion. Con un solo relevante en posicion 1, $nDCG@k = 1.0$.

### MAP (Mean Average Precision)

$$AP = \frac{1}{|relevant|} \sum_{k=1}^{|relevant|} P@k \cdot rel_k$$
$$MAP = \frac{1}{|Q|} \sum_{q \in Q} AP_q$$

Promedio de la precision en cada posicion donde aparece un relevante.

### MRR (Mean Reciprocal Rank)

$$MRR = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{rank_q}$$

Inversa del rango del primer relevante. Con relevante en posicion 1, $MRR = 1.0$.

## Resultados

### Tiempo de ajuste (5 corridas)

| Algoritmo | Media (s) | Std (s) | Min (s) | Max (s) |
|-----------|-----------|---------|---------|---------|
| TF-IDF | 0.3237 | 0.0143 | 0.31 | 0.34 |
| BB-IDF | 0.6279 | 0.0224 | 0.60 | 0.66 |
| TextRank | 7.2732 | 0.1036 | 7.16 | 7.40 |

TextRank es ~22x mas lento que TF-IDF y ~12x mas lento que BB-IDF.

### Tiempo de transformacion

| Algoritmo | Media (s) | Std (s) |
|-----------|-----------|---------|
| TF-IDF | 0.0392 | 0.0042 |
| BB-IDF | 0.0384 | 0.0026 |
| TextRank | 1.1587 | 0.0330 |

TextRank es ~30x mas lento en inferencia que TF-IDF y BB-IDF.

### Sparsity y Memoria

| Algoritmo | Vocab | Sparsity | Memoria (KB) |
|-----------|-------|----------|--------------|
| TF-IDF | 21,330 | 0.9092 | 5,665.78 |
| BB-IDF | 21,330 | 0.9092 | 5,665.78 |
| TextRank | 24,636 | 0.9181 | 6,543.94 |

TextRank produce un vocabulario ~15% mayor (incluye terminos con puntuacion TextRank > 0). TF-IDF y BB-IDF comparten el mismo vocabulario y matriz (CountVectorizer base).

### Metricas de recuperacion

| Algoritmo | P@5 | R@5 | nDCG@5 | MAP | MRR |
|-----------|-----|-----|--------|-----|-----|
| TF-IDF | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 |
| TextRank | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 |
| BB-IDF | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 |

Identicos. Ver seccion de metricas identicas.

### Escalabilidad

| n_docs | TF-IDF (s) | TextRank (s) | BB-IDF (s) |
|--------|------------|--------------|------------|
| 5 | 0.0741 | 1.1742 | 0.1683 |
| 10 | 0.1525 | 1.7430 | 0.3025 |
| 20 | 0.3092 | 3.9089 | 0.4983 |
| 34 | 0.5167 | 7.9811 | 0.8218 |

TF-IDF y BB-IDF escalan linealmente con el numero de documentos. TextRank tiene crecimiento super-lineal (componente $O(n^2)$ en construccion de grafo de co-ocurrencia).

### Analisis estadistico

- **Normalidad**: Los 3 algoritmos pasan Shapiro-Wilk (todos p > 0.05)
- **ANOVA**: $F = 20222.85$, $p < 0.001$, $\eta_p^2 = 0.9997$ (diferencias altamente significativas)
- **Post-hoc (Mann-Whitney + Cohen's d)**:
  - TF-IDF vs TextRank: $U=25$, $p=0.008$, $d=93.96$
  - BB-IDF vs TextRank: $U=25$, $p=0.008$, $d=88.66$
  - TF-IDF vs BB-IDF: $U=0$, $p=0.008$, $d=-16.19$

Todas las diferencias son significativas. El tamano del efecto (Cohen's d) es extremadamente grande ($|d| > 16$) en todos los pares, indicando que los algoritmos tienen rendimientos fundamentalmente distintos en tiempo de ejecucion.

## Graficos

### Informativos (diferencian algoritmos)

| Grafico | Descripcion |
|---------|-------------|
| `benchmark_fit_time.png` | Tiempo de ajuste por algoritmo. TextRank domina el grafico (~7.3s vs ~0.3-0.6s). |
| `benchmark_transform_time.png` | Tiempo de transformacion. TextRank nuevamente ~30x mas lento. |
| `benchmark_query_time.png` | Tiempo de consulta (similitud coseno). Similar para los 3. |
| `benchmark_vocab_size.png` | Tamanio del vocabulario generado por cada algoritmo. |
| `benchmark_panel_times.png` | Panel combinado con fit_time + vocab_size. |
| `benchmark_sparsity_density.png` | Sparsity y density de cada matriz. |
| `benchmark_memory.png` | Memoria de matriz y peso serializado. |
| `similarity_{algo}.png` | Heatmap de similitud coseno (5 consultas x 34 docs). Revela diferencias en la distribucion de similitudes. |
| `weight_distribution_{algo}.png` | Histograma + boxplot de la distribucion de pesos. Muestra como cada algoritmo distribuye los valores. |

### Identicos para los 3 algoritmos

| Grafico | Explicacion |
|---------|-------------|
| `precision_at_k.png` | P@k = {1.0, 1.0, 0.2, 0.2} para k={1,3,5,10}. Identico porque la consulta $q_i$ es el documento $d_i$, y $sim(d_i, d_i) = 1.0$ para los 3 algoritmos. El relevante siempre aparece en rank 1. |
| `recall_at_k.png` | R@k = 1.0 para todo k >= 1 por la misma razon: el unico relevante siempre se recupera. |
| `ndcg_at_k.png` | nDCG@k = 1.0 porque el relevante esta en posicion 1 (DCG = IDCG). |
| `mean_average_precision.png` | MAP = 1.0 porque precision en la unica posicion relevante (rank 1) es 1.0. |
| `mean_reciprocal_rank.png` | MRR = 1.0 porque el primer relevante siempre esta en rank 1. |
| `ranking_panel.png` | Panel combinado MAP + MRR, ambos 1.0. |

**Nota**: Estas metricas serian diferentes entre algoritmos si se usaran juicios de relevancia externos (ej. consultas fuera del corpus, evaluacion por un experto). Con la configuracion actual (query = doc), solo miden consistencia interna.

## Graficos que desestimamos

Los 6 graficos de metricas de recuperacion (`precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mean_average_precision`, `mean_reciprocal_rank`, `ranking_panel`) muestran valores identicos para los 3 algoritmos debido al diseno del ground truth artificial. Se mantienen en el codigo para uso futuro con juicios de relevancia reales, pero los consideramos no informativos en la evaluacion actual.

## Referencias

- Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into Text. *Proceedings of EMNLP*.
- Ramos, J. (2003). Using TF-IDF to Determine Word Relevance in Document Queries. *Proceedings of the First Instructional Conference on Machine Learning*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
