# Informe experimental: evaluación de BB-IDF frente a TF-IDF y TextRank

Evaluación de extracción de keywords sobre un corpus de documentos académicos
de turismo. Comparación principal **TF-IDF vs BB-IDF**, con **TextRank** como
referencia/benchmark.

---

## 1. Resumen del experimento

Se validaron las tres implementaciones y se estableció un protocolo
reproducible de extracción de keywords, evaluado contra las **palabras clave
declaradas por los autores** de cada documento (gold standard real, no
inventado).

**Hallazgo principal:** BB-IDF —definido como TF-IDF donde únicamente el conteo
de frecuencia documental `df(t)` se reemplaza por un `df` filtrado por banda
estadística— mejora a TF-IDF de forma **modesta y acotada**: la mejora es
**nominalmente** significativa en **F1@10 y R@10** (p = 0.006, d ≈ 0.56), pero
**no sobrevive a la corrección de Holm** por las 18 métricas exploradas
(p_adj = 0.105), y no es significativa en K = 5, 20 o 50, ni en MAP/MRR. BB-IDF
queda **a la par de TextRank** (sin diferencias significativas) y, en F1@10, lo
iguala o supera ligeramente. Además se detectó que la variante de BB-IDF con
**filtro duro de banda** (anular los pesos fuera de banda, conservada solo para
el análisis de robustez en `scorers.py::bbidf_weights_hard`) produce resultados
**catastróficos** (F1@10 ≈ 0.04), porque los umbrales de banda anulan ~91% de
los términos en documentos largos.

---

## 2. Corpus

- **N = 33 documentos PDF** (tesis y artículos de turismo; región
  Amazonas/Chachapoyas, Perú). El subdirectorio `data/corpus/resumenes/`
  (33 resúmenes en PDF) **no se procesa**: `load_documents` usa `glob("*.pdf")`
  no recursivo.
- **30/33** documentos declaran "Palabras clave(s)"/"Keywords" del autor.
  3 documentos sin keywords (`La (re)construcción de Chachapoyas.pdf`,
  `PERCEPCION DEL DESTINO CHACHAPOYAS.pdf`, `Planeamiento Estratégico...pdf`)
  → excluidos del gold standard.
- **30 documentos** forman el conjunto de evaluación (30 con keywords).
- Longitud: media 5 461 tokens, mediana 5 195, rango 879–15 782 (tras
  preprocesado).
- Gold: media 6.5 términos por documento (mediana 6, rango 4–14).
- Contenido multilingüe: 1 artículo íntegro en inglés (`Public Management of
  Tourism...`) y varios con resúmenes/revistas bilingües.

---

## 3. Preprocesamiento (idéntico para los 3 algoritmos y para el gold)

1. Extracción de texto: PyMuPDF (`fitz`), página a página.
2. Tokenización + lematización: spaCy `es_core_news_sm` (parser/ner
   deshabilitados), se usa `lemma_.lower()`.
3. Filtros: sin puntuación, sin espacios, sin stopwords (es + lista mínima de
   stopwords EN por el contenido bilingüe), sin tokens numéricos
   (`like_num`), **lema estrictamente alfabético** (`isalpha`, elimina
   ISSN/DOI/URLs/`S/.`), y longitud original ≥ 3.
4. **Vocabulario compartido**: unión ordenada de los tokens de los 33 docs
   (V = 15 953). Los tres algoritmos operan **sobre el mismo vocabulario y la
   misma matriz de conteos**, garantizando comparabilidad.

**Gold standard:** las keywords del autor se normalizan con el mismo pipeline y
se aplanan a unigramas de contenido. Política de idioma: se usa el idioma del
cuerpo del documento (el doc en inglés conserva keywords en inglés; un doc con
cuerpo en español y keywords solo en inglés fue traducido fielmente, ver
`bb_idf/experiment/gold.py`).

---

## 4. Implementación de TF-IDF

- `bb_idf/algorithms/tfidf.py` envuelve `sklearn.TfidfVectorizer` con
  `smooth_idf=True`, `norm='l2'`, `sublinear_tf=False`.
- En el experimento se usa la siguiente formulación:
  `idf(t) = ln((1+N)/(1+df(t))) + 1`, `w(t,d) = tf(t,d)·idf(t)`.
- **Correcta** conceptualmente. El módulo empaquetado (L2 + smooth_idf de
  sklearn) es equivalente en el ranking por documento; para la comparación
  controlada se usa la formulación descrita (ver §7).

## 5. Implementación de BB-IDF

- Nombre: **Bounded Band Inverse Document Frequency (BB-IDF)** — "IDF de Banda
  Acotada" (según la propuesta del proyecto).
- Fórmula implementada (fiel a la propuesta):
  - Banda por documento: `[μ_d + 0.5·σ_d, μ_d + 2.5·σ_d]`, con `μ_d, σ_d`
    sobre las frecuencias **no nulas** de los términos del documento `d`
    (σ poblacional).
  - Fallback `[1.5, 4.5]` si `banda_inf ≥ banda_sup` o el doc tiene < 30
    tokens.
  - `df_banda(t) = |{d : banda_inf_d ≤ f(t,d) ≤ banda_sup_d}|`.
  - `idf_bb(t) = ln((1+N)/(1+df_banda(t))) + 1` (misma fórmula que TF-IDF).
  - `w(t,d) = tf(t,d)·idf_bb(t)`.
- **Variables:** `f(t,d)` (conteo crudo), `N`, `μ_d`, `σ_d`, `df_banda`.
- **Diferencia vs TF-IDF:** solo el conteo de `df` (el "filtro de banda").
- **Comportamiento observado:** en documentos largos, `μ+0.5σ` sobre conteos
  crudos es muy alto (3.8–19.2), por lo que **91.2% de los términos tienen
  `df_banda = 0`**. El IDF de banda colapsa a casi-constante (std 0.33 vs 0.67
  del IDF clásico; correlación IDF-clásico/IDF-banda = 0.59). En consecuencia,
  BB-IDF (solo-df) se comporta como *ranking por frecuencia cruda* con una
  ligera penalización a términos muy frecuentes en muchas bandas.

## 6. Implementación de TextRank

- `bb_idf/algorithms/textrank.py`: grafo de co-ocurrencia por documento +
  PageRank ponderado (fórmula de Mihalcea & Tarau 2004), `(1-d)` como base.
- **Ventana de co-ocurrencia** `N = 2` (valor que reporta mejores resultados en
  M&T 2004). La sensibilidad a este parámetro se evalúa en robustez con
  ventanas 2, 5 y 10.
- En el experimento, TextRank opera **sobre el mismo vocabulario compartido**
  (puntuación por documento, 0 para términos ausentes).

---

## 7. Protocolo experimental

| Elemento | Valor |
|---|---|
| Unidad experimental | documento (n = 30 evaluables) |
| Algoritmos | TF-IDF, BB-IDF (solo cambio de `df`), TextRank (ventana 2) |
| Representación | matriz de conteos compartida N×V |
| Selección de keywords | Top-K por peso descendente (desempate por orden alfabético) |
| K | 5, 10, 20, 50 |
| Métricas | P@K, R@K, F1@K, AP, MRR, nDCG@K |
| Comparación principal | BB-IDF vs TF-IDF (pareada por documento; desenlace primario F1@10) |
| Comparación secundaria | BB-IDF/TF-IDF vs TextRank (benchmark) |
| Estadística | Wilcoxon signed-rank (pareado), bootstrap CI 95%, Cohen's d (pareado), rank-biserial, Holm-Bonferroni |
| Eficiencia | tiempo, throughput y memoria pico (tracemalloc), media ± std sobre N = 10 repeticiones |
| Semilla | 0 (bootstrap) |
| Salida | `results/{raw,processed,metrics,statistical,figures,tables}` |

**Decisiones metodológicas tomadas:**
1. **Aislar el efecto del filtro de banda**: BB-IDF difiere de TF-IDF solo en
   `df → df_banda` (misma fórmula IDF y mismo TF). De este modo, cualquier
   diferencia observada es atribuible al filtro de banda y no a cambios en la
   fórmula de IDF o en la normalización de TF.
2. **Gold = keywords de autores** (disponibles en 30/33 PDFs), en lugar de un
   ground truth de recuperación de documentos que no discriminaba entre
   algoritmos.
3. **Exclusión** de 3 documentos sin keywords declaradas (`La (re)construcción
   de Chachapoyas.pdf`, `PERCEPCION DEL DESTINO CHACHAPOYAS.pdf`,
   `Planeamiento Estratégico...pdf`), quedando 30 evaluables.
4. **nDCG@K** se reporta en los datos crudos; al ser relevancia binaria y
   pocos relevantes, es casi redundante con P@K y no se usa en conclusiones.

---

## 8. Métricas utilizadas

**Definición de `K`.** `K` es el número de palabras clave extraídas **por
documento** (Top-K). No se refiere al número de documentos (N = 33). Cada
algoritmo ordena los términos del documento por score descendente y conserva los
primeros `K`. Se evalúa K ∈ {5, 10, 20, 50}; dado que el gold tiene en promedio
~6.5 keywords por documento, K = 10 es el desenlace primario.

- **P@K**: fracción del Top-K que es keyword de autor.
- **R@K**: fracción de keywords de autor recuperadas en el Top-K.
- **F1@K**: media armónica de P@K y R@K.
- **AP (Average Precision)**: media de la precisión en cada posición de una
  keyword relevante (resume el ranking completo).
- **MAP (Mean Average Precision)**: media de AP sobre los documentos.
- **MRR**: inversa del rango de la primera keyword relevante.
- **nDCG@K**: DCG normalizado con relevancia binaria.

Son apropiadas para un gold standard en forma de **conjunto de términos**
(extracción de keywords). AP, MAP y MRR son válidos; nDCG@K con relevancia
binaria aporta poco más que P@K y no se usa en las conclusiones.

**Comparabilidad de scores.** Los scores crudos de los tres algoritmos viven en
escalas distintas (TF-IDF y BB-IDF son `tf × idf`, del orden de cientos;
TextRank es un valor de PageRank, del orden de decenas). Para permitir la
comparación entre algoritmos, todos los scores exportados se **normalizan con
L2 por documento** (`score = w / ‖w_d‖₂`, en [0, 1]), conservando el valor crudo
como `score_raw`. Esta normalización es una transformación monótona por
documento: **no altera el ranking ni las métricas**, que son invariantes a
escala.

---

## 9–11. Resultados (media F1@K y ranking)

| Métrica | TF-IDF | BB-IDF | TextRank |
|---|---|---|---|
| F1@5 | 0.432 | 0.442 | **0.455** |
| F1@10 | 0.437 | **0.477** | 0.467 |
| F1@20 | 0.348 | 0.347 | 0.349 |
| F1@50 | 0.190 | 0.190 | 0.185 |
| MAP | 0.488 | 0.500 | **0.508** |
| MRR | 0.741 | 0.747 | **0.828** |

(R@K sigue la misma pauta: R@10 = 0.587 / 0.639 / 0.631.)

## 12. Mejora de BB-IDF respecto a TF-IDF

Mejora media (%) = ((M_BB − M_TF)/M_TF)×100, por documento:

| Métrica | Mejora media | Mediana | Std | Min | Max |
|---|---|---|---|---|---|
| F1@5 | +3.4% | 0.0% | 27.1 | −50 | +100 |
| F1@10 | **+10.3%** | 0.0% | 18.2 | −25 | +50 |
| F1@20 | −0.9% | 0.0% | 11.4 | −50 | +25 |
| F1@50 | +0.5% | 0.0% | 7.8 | −20 | +33 |

En F1@10, BB-IDF gana en 10/30 documentos, pierde en 1 y empata en 19. La
mejora media positiva se debe a un subconjunto de documentos; la **mediana es
0** (dominan los empates), lo que indica que la ventaja es **no uniforme**.

## 13. Comparación contra TextRank (benchmark)

- **BB-IDF vs TextRank**: ninguna diferencia significativa en ninguna métrica
  (todos p > 0.2). En F1@10, BB-IDF (0.477) ≥ TextRank (0.467); en MAP, TextRank
  (0.508) > BB-IDF (0.500); en MRR, TextRank (0.828) > BB-IDF (0.747).
- **TF-IDF vs TextRank**: TextRank supera a TF-IDF en F1@5 (+0.023), MAP
  (+0.020) y MRR (+0.087); empate en F1@10/F1@20.
- **Porcentaje del desempeño de TextRank alcanzado por BB-IDF**: F1@10 = 102%
  (supera), MAP = 98%, MRR = 90%.
- BB-IDF supera a TextRank en 7/30 documentos (F1@10); TextRank supera a
  BB-IDF en 6/30; empate en 17/30.

## 14. Significancia estadística (BB-IDF vs TF-IDF)

Prueba pareada de Wilcoxon sobre las diferencias por documento:

La **unidad de análisis** es el documento (n = 30); las comparaciones son
**pareadas** porque cada documento aporta un valor para ambos algoritmos. Se
emplea **Wilcoxon signed-rank** (no paramétrico) porque las métricas por
documento no siguen una distribución normal (valores acotados y muchos empates).
El **p-value** indica la probabilidad de observar la diferencia bajo la hipótesis
nula (α = 0.05); el **tamaño del efecto** se reporta con **Cohen's d** pareado
(0.2 pequeño, 0.5 medio, 0.8 grande) y el **IC 95%** se obtiene por bootstrap.

| Métrica | p | p (Holm) | Cohen's d | n pares (dif ≠ 0) | IC 95% diff |
|---|---|---|---|---|---|
| F1@10 | **0.006** | 0.105 | +0.56 | 11 | [+0.016, +0.065] |
| R@10 | **0.006** | 0.105 | +0.56 | 11 | [+0.021, +0.085] |
| P@10 | 0.025 | 0.406 | +0.55 | 11 | [+0.013, +0.053] |
| F1@5 | 0.266 | 1.0 | +0.10 | 9 | [−0.025, +0.047] |
| F1@20 | 0.750 | 1.0 | −0.04 | 4 | — |
| F1@50 | 0.625 | 1.0 | +0.02 | 4 | — |
| MAP | 0.584 | 1.0 | +0.16 | 30 | [−0.013, +0.037] |
| MRR | 0.899 | 1.0 | +0.03 | 14 | — |

**Interpretación:** la mejora es **nominalmente** significativa **solo en
K=10** (P@10, R@10, F1@10). Sin embargo, al corregir por las **18 métricas**
exploradas de la comparación principal (Holm-Bonferroni), **ninguna sobrevive**:
F1@10 y R@10 pasan a p_adj = 0.105 (≥ 0.05) y P@10 a p_adj = 0.406. Es decir,
la aparente mejora en K=10 **no es estadísticamente significativa tras el
control por comparaciones múltiples**. El bajo número de pares no empatados
(11 de 30) y la dependencia del resultado respecto de K refuerzan que la
conclusión es **exploratoria**.

## 15. Tamaño del efecto

- Cohen's d (pareado) en F1@10 = **0.56** (efecto **moderado**, umbrales
  habituales: 0.2 pequeño, 0.5 medio, 0.8 grande).
- Rank-biserial en F1@10 ≈ +1.0 entre los pares que difieren (la dirección es
  consistente: donde hay diferencia, BB-IDF casi siempre gana).
- El efecto no es uniforme: concentrado en ~1/3 de los documentos.

## 16. Eficiencia (media ± std, N = 10 repeticiones)

| Métrica | TF-IDF | BB-IDF | TextRank |
|---|---|---|---|
| Tiempo total | 0.0054 ± 0.0005 s | 0.0095 ± 0.0009 s | 3.99 ± 0.02 s |
| Throughput | ~6 100 docs/s | ~3 500 docs/s | ~8 docs/s |
| Memoria pico (tracemalloc) | ~8.5 MB | ~8.5 MB | ~102 MB |
| Factor tiempo vs TF-IDF | 1× | **~1.8×** | **~740×** |

(Preprocesado spaCy: ~60 s, único, compartido y cacheado; no incluido en los
tiempos. El tiempo por documento y las series por documento están en
`results/raw/per_doc_times.csv` y en cada `results/per_document/docX/timing.csv`.)
BB-IDF cuesta ~1.8× TF-IDF (cálculo de banda y conteo filtrado); TextRank es
~740× más caro y ~12× más memoria (grafo de co-ocurrencia por documento).
El análisis de complejidad asintótica está en
[docs/analisis_complejidad.md](docs/analisis_complejidad.md): TF-IDF y BB-IDF
son O(N·V); TextRank es O(N·(L̄·W + I·Ū²)).

## 16b. Similitud de ranking con TextRank (análisis suplementario)

Complementariamente a la calidad, se cuantifica cuánto se **parece** el ranking
de cada algoritmo al de TextRank (medida de *convergencia de rankings*, no de
calidad). Métricas: **RBO** (Rank-Biased Overlap, p=0.9), **Jaccard@K** y
**Overlap@K**, por documento.

| Métrica | TF-IDF | BB-IDF | p (pareado) | Cohen's d |
|---|---|---|---|---|
| RBO@10 | 0.615 | **0.705** | 0.0002 | +0.71 |
| Jaccard@10 | 0.594 | **0.686** | 0.0003 | +0.86 |
| Overlap@10 | 0.727 | **0.800** | 0.0006 | +0.78 |

**BB-IDF es consistentemente más similar a TextRank que TF-IDF**, en todas las
métricas y en todos los K (significativo en K ≥ 10, p < 0.005; en K = 5 solo RBO
es significativo). Este resultado es coherente con la interpretación principal:
al colapsar el IDF hacia casi-constante, BB-IDF se comporta como un ranking por
frecuencia, más próximo al comportamiento de TextRank que al de TF-IDF.

## 17. Visualizaciones (`results/figures/`)

1. `prf_at_k.png` — P/R/F1@K de los 3 algoritmos (valores anotados).
2. `improvement_bbidf.png` — mejora % de BB-IDF vs TF-IDF (media ± SE).
3. `per_doc_boxplot.png` — distribución por documento (F1@10, AP) con puntos individuales.
4. `efficiency.png` — tiempo total, tiempo/doc, memoria pico y throughput (panel 2×2).
5. `quality_time_tradeoff.png` — F1@10 vs tiempo (escala log).
6. `heatmap_f1_doc.png` — F1@10 por documento × algoritmo (valores en celdas).
7. `f1_with_ci.png` — medias F1@K con IC 95%.
8. `paired_diff_f1.png` — histograma de diferencias pareadas (BB-IDF − TF-IDF).
9. `wordclouds.png` — **nubes de palabras** de cada algoritmo (score acumulado).
10. `similarity_to_textrank.png` — similitud de ranking con TextRank (RBO) de TF-IDF vs BB-IDF.
11. `similarity_per_doc.png` — RBO@50 y Jaccard@50 vs TextRank por documento.
12. `gap_to_textrank_per_doc.png` — gap de F1@10 a TextRank por documento.

Las nubes de palabras (`wordclouds.png`) permiten la comparación cualitativa
visual de qué términos prioriza cada método: TF-IDF tiende a términos
discriminativos/raros, mientras BB-IDF y TextRank favorecen términos frecuentes
más alineados con las keywords de los autores.

---

## 18. Análisis cualitativo (casos)

Ver `results/tables/case_analysis.md`. Ejemplos:

- **Mayor ventaja de BB-IDF** (doc 20, "Valoración económica de Kuélap"): gold
  = {valoración, económica, complejo, arqueológico, costo, viaje, excedente,
  consumidor, individual}. F1@10: TF-IDF 0.526 → BB-IDF 0.737. BB-IDF rescata
  "viaje", "complejo", "consumidor" que TF-IDF relega.
- **Peor desempeño de BB-IDF** (doc 3, artículo bilingüe): gold con términos
  genéricos ("turismo", "actividad turística"); BB-IDF empeora por el ruido
  bilingüe ("negonotas", "revista", "the").
- **Donde BB-IDF supera a TextRank** (doc 5, 6, 8, 17, 18, 19, 20): TextRank
  penaliza términos específicos de un documento único (ej. "kuelap",
  "satisfacción") que TF-IDF/BB-IDF sí destacan.
- **Documentos difíciles para todos** (doc 1 "Chachapoyas Resort", un plan de
  negocios): el contenido real es financiero ("costo", "caja", "tabla"),
  mientras las keywords del autor son conceptuales ("turismo, experiencias,
  cultura...") → F1 ≈ 0.13 para los tres.

**Características que explican diferencias:** (a) especificidad del tema: los
métodos basados en frecuencia (BB-IDF, TextRank) favorecen términos frecuentes,
que coinciden más con keywords de autor genéricas; TF-IDF favorece términos
discriminativos raros, a veces ausentes de las keywords. (b) bilingüismo. (c)
documentos con contenido no temático (planes de negocio, metodología).

## 19. Limitaciones

1. **n = 30** documentos: potencia limitada; conclusiones exploratorias.
2. **Gold de unigramas**: las keywords del autor son frases multi-palabra; se
   aplanaron a unigramas, perdiendo la estructura de frases (los algoritmos
   son unigrama). No se evaluó la exactitud de frase completa.
3. **Lematizador español sobre texto inglés**: el artículo en inglés (y los
   fragmentos bilingües) no se lematizan bien ("tourists" ≠ "tourist"),
   penalizando ese documento por igual a los 3 métodos.
4. **Empates masivos**: BB-IDF y TF-IDF empatan en ~19/30 docs en F1@10; la
   mejora es real solo en un subconjunto.
5. **La mejora depende de K y no sobrevive a Holm**: nominalmente significativa
   solo en K=10 (p = 0.006 → p_adj = 0.105). No es un resultado robusto.
6. **El filtro duro de banda es inviable** con umbrales actuales (ver §20).
7. El corpus consta de 33 documentos (30 evaluables); el subdirectorio
   `data/corpus/resumenes/` no se procesa.
8. No hay anotación humana independiente (el gold es el de los autores, que es
   la referencia más defendible disponible, pero no un juicio de expertos).

## 20. Conclusiones

**Resultados observados (con 30 documentos):**
- BB-IDF (solo-df) mejora a TF-IDF en **F1@10: 0.437 → 0.477 (+10.3%)**, y en
  R@10/P@10, de forma **nominalmente significativa** (p = 0.006, d = 0.56),
  pero **no sobrevive a la corrección de Holm** (p_adj = 0.105).
- La mejora **no es significativa** en K = 5, 20, 50, ni en MAP/MRR.
- BB-IDF está **a la par de TextRank** (sin diferencias significativas), y lo
  supera ligeramente en F1@10.
- La variante de BB-IDF con **filtro duro de banda** (anular los pesos fuera de
  banda) colapsa (F1@10 ≈ 0.04) porque los umbrales anulan ~91% de términos.

**Respuestas directas:**
1. **¿BB-IDF mejora a TF-IDF?** Sí, pero **modestamente** y **solo en K=10**
   (y en R@10/P@10); en el resto de configuraciones es indistinguible.
2. **¿Cuánto?** F1@10: +10.3% de media (de 0.437 a 0.477), con efecto
   moderado (d = 0.56). La mediana de mejora es 0% (muchos empates).
3. **¿Es significativa?** **Nominalmente** en F1@10/R@10 (p = 0.006), pero
   **no tras la corrección de Holm** (p_adj = 0.105); no en K=5/20/50/MAP/MRR.
4. **¿Es consistente?** No: gana en 10/30, empata en 19/30, pierde en 1/30.
   La ventaja se concentra en documentos con keywords específicas frecuentes.
5. **¿Qué tan cerca está BB-IDF de TextRank?** A la par; F1@10 = 102% del
   desempeño de TextRank, MAP = 98%, MRR = 90%.
6. **¿Costo computacional?** ~1.8× el de TF-IDF (0.0095 s vs 0.0054 s para 33
   docs, media de 10 corridas); ~420× más barato que TextRank (3.99 s).

**Interpretación:** el filtro de banda, aplicado *solo* al conteo de `df`,
equivale en la práctica a aplanar el IDF hacia casi-constante (91% de términos
con `df_banda = 0`), convirtiendo a BB-IDF en un ranking por frecuencia con
penalización selectiva. Esto lo acerca al comportamiento de TextRank y lo hace
ligeramente mejor que TF-IDF para keywords de autor *genéricas*, pero el efecto
es pequeño y dependiente de K.

**Qué NO puede sostenerse con 30 documentos:** que BB-IDF sea globalmente
superior a TF-IDF o a TextRank; cualquier extrapolación a otros dominios; que
la mejora sea grande o universal; que la mejora en K=10 sea estadísticamente
significativa tras corrección por comparaciones múltiples. La variante con
filtro duro (anular los pesos fuera de banda) **no** es superior: es inviable
con los umbrales de banda actuales, y debería descartarse.

---

## Archivos generados

- `results/raw/per_doc_metrics.csv` — métricas por documento/algoritmo/K.
- `results/raw/per_doc_times.csv` — tiempo de scoring por documento/algoritmo.
- `results/processed/summary.csv`, `per_doc_wide.csv` (una fila por documento),
  `improvement_bbidf_vs_tfidf.csv`, `gap_to_textrank.csv`.
- `results/statistical/paired_tests.csv` (con `p_holm`), `similarity_tests.csv`,
  `robustness.csv`, `band_diagnostic.csv`.
- `results/metrics/keywords_ranked.csv` (ranking completo con score),
  `top10_keywords.csv`, `top5_keywords.csv`, `top50_keywords.csv` (documento,
  algoritmo, rank, keyword, score).
- `results/tables/case_analysis.md`, `per_document.md` (ranking Top-10 por
  documento con gold resaltado).
- `results/per_document/doc1/…doc33/` — por documento: top-50 keywords por
  algoritmo (`tfidf.csv`, `bbidf.csv`, `textrank.csv`, con score L2 + raw),
  nube de palabras por algoritmo, `similarity_textrank.csv` (RBO/Jaccard/Overlap
  vs TextRank) y `timing.csv`; `documents_index.csv` mapea `docX` al archivo
  original.
- `results/figures/*.png` (12 figuras, incl. nubes de palabras, frecuencia y
  similitud con TextRank).
- `results/processed/similarity_to_textrank.csv` y
  `results/statistical/similarity_tests.csv` — análisis suplementario de
  similitud de ranking con TextRank.
- `results/benchmark/` — salida del benchmark computacional (`main.py`).
- `results/metadata.json` — configuración, tiempos/memoria (media ± std) y
  throughput.
- `docs/analisis_complejidad.md` — análisis de complejidad Big-O comparativo.
