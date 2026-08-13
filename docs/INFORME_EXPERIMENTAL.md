# Informe experimental: evaluación de BB-IDF frente a TF-IDF y TextRank

Evaluación de extracción de keywords sobre un corpus de documentos académicos
de turismo. Comparación principal **TF-IDF vs BB-IDF**, con **TextRank** como
referencia/benchmark.

---

## 1. Resumen del experimento

Se auditaron las tres implementaciones del proyecto, se corrigió el diseño
experimental (que no medía calidad) y se diseñó un protocolo reproducible de
extracción de keywords evaluado contra las **palabras clave declaradas por los
autores** de cada documento (gold standard real, no inventado).

**Hallazgo principal:** BB-IDF —definido como TF-IDF donde únicamente el conteo
de frecuencia documental `df(t)` se reemplaza por un `df` filtrado por banda
estadística— mejora a TF-IDF de forma **modesta y acotada**: la mejora es
estadísticamente significativa en **F1@10 y R@10** (p = 0.006, d ≈ 0.56), pero
no significativa en K = 5, 20 o 50, ni en AP/MRR. BB-IDF queda **a la par de
TextRank** (sin diferencias significativas) y, en F1@10, lo iguala o supera
ligeramente. Además se detectó que la variante de BB-IDF con **filtro duro de
banda** (la que implementa `bb_idf/algorithms/bbidf.py`) produce resultados
**catastróficos** (F1@10 ≈ 0.04), porque los umbrales de banda anulan ~91% de
los términos en documentos largos.

---

## 2. Corpus

- **N = 33 documentos PDF** (no 34 como declaran README y `queries.json`; un
  PDF falta). Documentos: tesis y artículos de turismo (mayormente región
  Amazonas/Chachapoyas, Perú).
- **31/33** documentos declaran "Palabras clave(s)"/"Keywords" del autor.
  2 documentos sin keywords (`La (re)construcción de Chachapoyas.pdf`,
  `Planeamiento Estratégico...pdf`) → excluidos del gold standard.
- **30 documentos** forman el conjunto de evaluación (31 con keywords − 1
  excluido por keywords en inglés con cuerpo en español, ver §3).
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

## 4. Auditoría de TF-IDF

- `bb_idf/algorithms/tfidf.py` envuelve `sklearn.TfidfVectorizer` con
  `smooth_idf=True`, `norm='l2'`, `sublinear_tf=False`.
- En el experimento se usa la fórmula del proyecto (notebooks de referencia):
  `idf(t) = ln((1+N)/(1+df(t))) + 1`, `w(t,d) = tf(t,d)·idf(t)`.
- **Correcto** conceptualmente. La versión empaquetada difiere de la de los
  notebooks (L2 + smooth_idf de sklearn), lo que originalmente confundía la
  comparación (ver §7).

## 5. Auditoría de BB-IDF

- Nombre: **Bounded Band Inverse Document Frequency (BB-IDF)** — "IDF de Banda
  Acotada" (según `nuevo/BB-IDF Propuesta.docx`).
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

## 6. Auditoría de TextRank

- `bb_idf/algorithms/textrank.py`: grafo de co-ocurrencia por documento +
  PageRank ponderado (fórmula de Mihalcea & Tarau 2004), `(1-d)` como base.
- **Inconsistencia:** la ventana por defecto es `window_size=2` en el código,
  pero el README y el notebook declaran 5. El experimento usa ventana 2
  (coincide con el mejor resultado reportado por M&T) y evalúa 2/5/10 en
  robustez.
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
| Comparación principal | BB-IDF vs TF-IDF (pareada por documento) |
| Comparación secundaria | BB-IDF/TF-IDF vs TextRank (benchmark) |
| Estadística | Wilcoxon signed-rank (pareado), bootstrap CI 95%, Cohen's d (pareado), rank-biserial |
| Semilla | 0 (bootstrap) |
| Salida | `results/{raw,processed,metrics,statistical,figures,tables}` |

**Decisiones metodológicas tomadas (no definidas en el proyecto):**
1. **Aislar el efecto del filtro de banda**: BB-IDF difiere de TF-IDF solo en
   `df → df_banda` (misma fórmula IDF y mismo TF). Sin esto, la comparación
   estaba confundida por fórmulas IDF distintas (sklearn vs `ln(1+N/(df+1))`).
2. **Gold = keywords de autores** (disponibles en 31/33 PDFs), en vez del
   qrels degenerado `consulta = documento` que hacía todas las métricas = 1.0.
3. **Exclusión** de 3 documentos sin gold utilizable (2 sin keywords; 1 con
   keywords en inglés y cuerpo en español → se tradujo, quedando 30).
4. **nDCG@K** se reporta en los datos crudos; al ser relevancia binaria y
   pocos relevantes, es casi redundante con P@K y no se usa en conclusiones.

---

## 8. Métricas utilizadas

- **P@K**: fracción del Top-K que es keyword de autor.
- **R@K**: fracción de keywords de autor recuperadas en el Top-K.
- **F1@K**: media armónica de P@K y R@K.
- **AP (Average Precision)**: media de la precisión en cada posición de una
  keyword relevante (resume el ranking completo).
- **MRR**: inversa del rango de la primera keyword relevante.
- **nDCG@K**: DCG normalizado con relevancia binaria.

Son apropiadas para un gold standard en forma de **conjunto de términos**
(extracción de keywords). MAP/MRR/AP son válidos; nDCG@K con relevancia
binaria aporta poco más que P@K y no se usa en las conclusiones.

---

## 9–11. Resultados (media F1@K y ranking)

| Métrica | TF-IDF | BB-IDF | TextRank |
|---|---|---|---|
| F1@5 | 0.432 | 0.442 | **0.455** |
| F1@10 | 0.437 | **0.477** | 0.467 |
| F1@20 | 0.348 | 0.347 | 0.349 |
| F1@50 | 0.190 | 0.190 | 0.185 |
| AP | 0.488 | 0.500 | **0.508** |
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
  (todos p > 0.2). En F1@10, BB-IDF (0.477) ≥ TextRank (0.467); en AP, TextRank
  (0.508) > BB-IDF (0.500); en MRR, TextRank (0.828) > BB-IDF (0.747).
- **TF-IDF vs TextRank**: TextRank supera a TF-IDF en F1@5 (+0.023), AP
  (+0.020) y MRR (+0.087); empate en F1@10/F1@20.
- **Porcentaje del desempeño de TextRank alcanzado por BB-IDF**: F1@10 = 102%
  (supera), AP = 98%, MRR = 90%.
- BB-IDF supera a TextRank en 7/30 documentos (F1@10); TextRank supera a
  BB-IDF en 6/30; empate en 17/30.

## 14. Significancia estadística (BB-IDF vs TF-IDF)

Prueba pareada de Wilcoxon sobre las diferencias por documento:

| Métrica | p | Cohen's d | n pares (dif ≠ 0) | IC 95% diff |
|---|---|---|---|---|
| F1@10 | **0.006** | +0.56 | 11 | [+0.016, +0.065] |
| R@10 | **0.006** | +0.56 | 11 | [+0.021, +0.085] |
| P@10 | 0.025 | +0.55 | 11 | [+0.013, +0.053] |
| F1@5 | 0.266 | +0.10 | 9 | [−0.025, +0.047] |
| F1@20 | 0.750 | −0.04 | 4 | — |
| F1@50 | 0.625 | +0.02 | 4 | — |
| AP | 0.584 | +0.16 | 30 | [−0.013, +0.037] |
| MRR | 0.899 | +0.03 | 14 | — |

**Interpretación:** la mejora es estadísticamente significativa **solo en
K=10** (P@10, R@10, F1@10). Tratando F1@10 como desenlace primario, p = 0.006
(no requiere corrección). Si se corrigiera por las 10 métricas exploradas
(Holm), F1@10 y R@10 quedarían en el límite (p_adj ≈ 0.05); el resto no
sobrevive. El bajo número de pares no empatados en K=5/20/50 limita la
potencia (muchos empates).

## 15. Tamaño del efecto

- Cohen's d (pareado) en F1@10 = **0.56** (efecto **moderado**, umbrales
  habituales: 0.2 pequeño, 0.5 medio, 0.8 grande).
- Rank-biserial en F1@10 ≈ +1.0 entre los pares que difieren (la dirección es
  consistente: donde hay diferencia, BB-IDF casi siempre gana).
- El efecto no es uniforme: concentrado en ~1/3 de los documentos.

## 16. Eficiencia

| Métrica | TF-IDF | BB-IDF | TextRank |
|---|---|---|---|
| Tiempo total (s) | 0.0042 | 0.0089 | 4.117 |
| Tiempo/doc (s) | 0.00013 | 0.00027 | 0.125 |
| Factor vs TF-IDF | 1× | **2.1×** | **~980×** |

(Preprocesado spaCy: 60.1 s, único y compartido.) BB-IDF cuesta ~2× TF-IDF
(por el cálculo de banda y el conteo filtrado); TextRank es ~3 órdenes de
magnitud más caro. El trade-off calidad/costo favorece a BB-IDF frente a
TextRank.

## 17. Visualizaciones (`results/figures/`)

1. `prf_at_k.png` — P/R/F1@K de los 3 algoritmos.
2. `f1_comparison.png` — barras F1@K.
3. `improvement_bbidf.png` — mejora % de BB-IDF vs TF-IDF (con error estándar).
4. `per_doc_boxplot.png` — distribución por documento (F1@10, AP).
5. `efficiency.png` — tiempo total y por documento.
6. `quality_time_tradeoff.png` — F1@10 vs tiempo (escala log).
7. `heatmap_f1_doc.png` — F1@10 por documento × algoritmo.
8. `f1_with_ci.png` — medias F1@K con IC 95%.
9. `paired_diff_f1.png` — histograma de diferencias pareadas (BB-IDF − TF-IDF).

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
5. **La mejora depende de K**: significativa solo en K=10. No es un resultado
   robusto a todos los K.
6. **El filtro duro de banda es inviable** con umbrales actuales (ver §20).
7. Un documento del corpus falta (33 en vez de 34); el `queries.json` del
   proyecto no está alineado con el orden de carga real y no se usó.
8. No hay anotación humana independiente (el gold es el de los autores, que es
   la referencia más defendible disponible, pero no un juicio de expertos).

## 20. Conclusiones

**Resultados observados (con 30 documentos):**
- BB-IDF (solo-df) mejora a TF-IDF en **F1@10: 0.437 → 0.477 (+10.3%)**, y en
  R@10/P@10, de forma **estadísticamente significativa** (p = 0.006, d = 0.56).
- La mejora **no es significativa** en K = 5, 20, 50, ni en AP/MRR.
- BB-IDF está **a la par de TextRank** (sin diferencias significativas), y lo
  supera ligeramente en F1@10.
- La variante de BB-IDF con **filtro duro de banda** (la de `bbidf.py`)
  colapsa (F1@10 ≈ 0.04) porque los umbrales anulan ~91% de términos.

**Respuestas directas:**
1. **¿BB-IDF mejora a TF-IDF?** Sí, pero **modestamente** y **solo en K=10**
   (y en R@10/P@10); en el resto de configuraciones es indistinguible.
2. **¿Cuánto?** F1@10: +10.3% de media (de 0.437 a 0.477), con efecto
   moderado (d = 0.56). La mediana de mejora es 0% (muchos empates).
3. **¿Es significativa?** Sí en F1@10/R@10 (p = 0.006); no en K=5/20/50/AP/MRR.
4. **¿Es consistente?** No: gana en 10/30, empata en 19/30, pierde en 1/30.
   La ventaja se concentra en documentos con keywords específicas frecuentes.
5. **¿Qué tan cerca está BB-IDF de TextRank?** A la par; F1@10 = 102% del
   desempeño de TextRank, AP = 98%, MRR = 90%.
6. **¿Costo computacional?** ~2× el de TF-IDF (0.009 s vs 0.004 s para 33
   docs); ~3 órdenes de magnitud más barato que TextRank (4.1 s).

**Interpretación:** el filtro de banda, aplicado *solo* al conteo de `df`,
equivale en la práctica a aplanar el IDF hacia casi-constante (91% de términos
con `df_banda = 0`), convirtiendo a BB-IDF en un ranking por frecuencia con
penalización selectiva. Esto lo acerca al comportamiento de TextRank y lo hace
ligeramente mejor que TF-IDF para keywords de autor *genéricas*, pero el efecto
es pequeño y dependiente de K.

**Qué NO puede sostenerse con 30 documentos:** que BB-IDF sea globalmente
superior a TF-IDF o a TextRank; cualquier extrapolación a otros dominios; que
la mejora sea grande o universal. La versión con filtro duro (la implementada
en el paquete) **no** es superior: es inviable con los umbrales de banda
actuales, y debería revisarse antes de cualquier uso.

---

## Archivos generados

- `results/raw/per_doc_metrics.csv` — métricas por documento/algoritmo/K.
- `results/processed/summary.csv`, `improvement_bbidf_vs_tfidf.csv`,
  `gap_to_textrank.csv`.
- `results/statistical/paired_tests.csv`, `robustness.csv`,
  `band_diagnostic.csv`.
- `results/metrics/top10_keywords.csv`, `keywords_ranked.csv`.
- `results/tables/case_analysis.md`.
- `results/figures/*.png` (9 figuras).
- `results/metadata.json` — configuración reproducible.
