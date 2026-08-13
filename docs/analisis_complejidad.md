# Análisis de complejidad computacional (Big-O / asintótica)

Comparación de la complejidad temporal y espacial de **TF-IDF** (baseline),
**BB-IDF** (propuesta) y **TextRank** (benchmark). El objetivo es determinar si
BB-IDF conserva la eficiencia lineal de TF-IDF o introduce un coste asintótico
mayor, y cómo se compara con el coste cuadrático de TextRank.

---

## 1. Notación

| Símbolo | Significado |
|---|---|
| `N` | número de documentos del corpus |
| `V` | tamaño del vocabulario (términos únicos del corpus) |
| `L_d` | número de tokens del documento `d` (media `L̄`) |
| `U_d` | número de términos **únicos** del documento `d` (media `Ū`) |
| `W` | ventana de co-ocurrencia de TextRank |
| `I` | iteraciones de PageRank hasta convergencia |
| `K` | número de keywords extraídas (Top-K) |

Relaciones útiles: `U_d ≤ L_d`, y `V ≤ Σ_d U_d = N·Ū` (el vocabulario es la
unión; suele ser menor que `N·Ū` por solapamiento entre documentos).

---

## 2. TF-IDF (baseline)

Representación: matriz de conteos término–documento `X` (`N × V`).

1. Frecuencia documental `df(t) = Σ_d [f_{t,d} > 0]`: **O(N·V)**.
2. `idf(t)` para los `V` términos: **O(V)**.
3. Pesos `w_{t,d} = f_{t,d} · idf(t)`: **O(N·V)**.

| | Complejidad |
|---|---|
| **Tiempo** | **O(N·V)** (denso) · **O(N·Ū)** (disperso, `nnz`) |
| **Espacio** | **O(N·V)** (denso) · **O(N·Ū)** (disperso) |

TF-IDF es **lineal** en el número de documentos y en el tamaño del vocabulario.

---

## 3. BB-IDF (propuesta)

BB-IDF añade el **filtro de banda** al conteo de `df`, sin cambiar la estructura
de la matriz:

1. Banda por documento: `μ_d, σ_d` sobre las frecuencias no nulas de `d`:
   **O(U_d)** por documento → **O(N·Ū)** en total.
2. `df_banda(t)` = conteo de documentos donde `f_{t,d}` cae dentro de la banda
   de ese documento: **O(N·V)** (denso) · **O(N·Ū)** (disperso).
3. `idf_banda(t)`: **O(V)**.
4. Pesos `w_{t,d} = f_{t,d} · idf_banda(t)`: **O(N·V)**.

| | Complejidad |
|---|---|
| **Tiempo** | **O(N·V + N·Ū)** = **O(N·V)** (denso) |
| **Espacio** | **O(N·V)** |

**Conclusión asintótica:** BB-IDF tiene el **mismo orden de complejidad que
TF-IDF** (lineal). El único coste añadido es un **factor constante** (el cálculo
de la banda `O(N·Ū)` y las comparaciones del `df_banda`), que se refleja en el
~2.1× medido experimentalmente.

---

## 4. TextRank (benchmark)

TextRank construye, **por documento**, un grafo de co-ocurrencia y ejecuta
PageRank. El grafo es una matriz de adyacencia `U_d × U_d` (nodos = términos
únicos del documento).

1. Construcción del grafo: para cada token se examinan `W` vecinos →
   **O(L_d · W)** por documento.
2. PageRank: `I` iteraciones de un producto matriz–vector `U_d × U_d` →
   **O(I · U_d²)** por documento (denso).
3. Extracción de keywords: ordenar `U_d` scores → **O(U_d log U_d)**.

| | Complejidad |
|---|---|
| **Tiempo** | **O(N · (L̄·W + I·Ū²))** |
| **Espacio** | **O(Ū²)** pico por documento |

**Conclusión asintótica:** TextRank es **cuadrático en el número de términos
únicos por documento** (`Ū²`). Como `Ū` crece con la longitud del documento, el
coste por documento crece de forma **superlineal**. Esto explica el ~980×
medido respecto a TF-IDF (y ~460× respecto a BB-IDF).

---

## 5. Tabla comparativa

| Método | Tiempo | Espacio | Crecimiento |
|---|---|---|---|
| TF-IDF | O(N·V) | O(N·V) | **lineal** |
| BB-IDF | O(N·V + N·Ū) | O(N·V) | **lineal** (constante mayor) |
| TextRank | O(N·(L̄·W + I·Ū²)) | O(Ū²) pico | **cuadrático en Ū** |

---

## 6. Confirmación empírica (33 documentos)

| Algoritmo | Tiempo total | Tiempo/doc | Factor vs TF-IDF |
|---|---|---|---|
| TF-IDF | 0.0042 s | 0.00013 s | 1× |
| BB-IDF | 0.0089 s | 0.00027 s | ~2.1× |
| TextRank | 4.117 s | 0.125 s | ~980× |

El factor ~2.1× de BB-IDF corresponde al **overhead constante** del filtro de
banda, coherente con el mismo orden O(N·V). El factor ~980× de TextRank
corresponde a su coste **cuadrático por documento**.

> Nota: el preprocesado (spaCy) es O(N·L̄) y es **compartido** por los tres
> métodos, por lo que no forma parte de la complejidad específica de cada
> algoritmo (y no se incluye en los tiempos anteriores).

---

## 7. Conclusión

BB-IDF **no introduce coste asintótico adicional** frente a TF-IDF: ambos son
lineales en `N` y `V`; la única diferencia es un factor constante pequeño
(~2×). En cambio, TextRank paga un coste **cuadrático por documento** (grafo de
co-ocurrencia + PageRank), lo que lo hace varios órdenes de magnitud más caro
en documentos largos. Por tanto, en términos de complejidad, BB-IDF conserva la
escalabilidad de TF-IDF y es asintóticamente mucho más eficiente que TextRank.
