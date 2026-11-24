# Modelo B — Metadatos y texto

Clustering de los mismos juegos del modelo A, añadiendo reseñas en inglés.
Metacritic y SteamSpy se reservan para la validación externa.

## Seguimiento

- [x] Crear la carpeta y la documentación.
- [x] Definir cómo seleccionar las reseñas y recopilar las reseñas de la muestra
  (consulta el historial del proyecto para el script de descarga utilizado).
- [x] Descargar 100 reseñas en inglés por juego (2.000 registros válidos).
- [x] Auditar la cobertura y fijar el umbral de 10 reseñas (1.942 juegos).
- [x] Crear `prepare_features.py`: preparar la descripción y las reseñas,
  aplicar el umbral de 10 reseñas y combinarlas con los metadatos del
  modelo A.
- [x] Crear `cluster_metadata_text.py`: probar algoritmos y números de
  grupos sobre la muestra de 1.942 juegos.
- [x] Generar `A_sub`, comparar las asignaciones A × B con ARI y tabla cruzada
  mediante `scripts/compare_models.py`.

**Lo que sigue (en el README principal):** , figuras y redacción de conclusiones.

La comparación guarda las asignaciones alineadas, las asignaciones A_sub, la
tabla cruzada y el resumen reproducible en `modelos/comparacion_A_B/`.
El ARI resultante es -0,0018 para los 1.942 juegos comunes.

## Script `prepare_features.py`

Prepara la tabla numérica combinando metadatos y representaciones de texto; todavía no entrena ni asigna grupos:

1. Lee `data/interim/final_sample.csv` mediante `scripts/modelo_A/prepare_features.py`, comprueba que no esté vacío ni tenga identificadores inválidos o duplicados y ordena los juegos por `appid`.
2. Lee `data/raw/reviews.jsonl` para los `appid` de la muestra, conserva el último registro válido (`status == 200` y `query_summary` presente), limpia el texto de las reseñas y excluye los juegos que no alcanzan el mínimo de 10 reseñas con texto (`--min-reviews`). Después construye **un documento por juego** concatenando sus reseñas únicas (elimina duplicados exactos) y ajusta el TF-IDF sobre esos documentos, no sobre las reseñas sueltas; así `max_df` filtra los términos que aparecen en más de la mitad de los juegos.
3. Busca los metadatos de los juegos restantes en `data/raw/appdetails_candidates*.jsonl`, valida los datos y aplica las transformaciones de los cuatro bloques junto con su reescalado por varianza.
4. Guarda los resultados en `data/processed/modelo_B/` (features y datos de entrada) y `modelos/modelo_B/` (artefactos del clustering):
   - `features.csv`: una fila por juego incluido, con `appid` y las variables finales escaladas.
   - `preprocessing.json`: columnas, juegos excluidos por tener pocas reseñas, varianzas antes y después del escalado, pesos, parámetros de estandarización, tamaños de vocabulario, varianza explicada por SVD, campos de descripción utilizados, fecha de referencia, semilla y reglas aplicadas.
   - `text_vocabulary.json`: listas de términos del vocabulario TF-IDF para reseñas y descripciones.
   - `tfidf_reviews.npz` y `tfidf_description.npz`: matrices dispersas CSR con los vectores TF-IDF normalizados L2 por juego (previos a la proyección SVD).
   - `text_models.joblib`: diccionario serializado con los objetos `TfidfVectorizer` y `TruncatedSVD` ajustados para reseñas y descripciones.

A diferencia del modelo A, este script filtra explícitamente los juegos sin suficiente texto de reseñas y registra los descartados en `preprocessing.json`. Se detiene con error si faltan registros válidos de reseñas para algún juego de la muestra inicial, si ningún juego alcanza `--min-reviews`, si faltan metadatos o campos obligatorios del modelo A, si algún juego carece de descripción en los tres campos candidatos, si `query_summary` no incluye totales válidos de reseñas positivas y negativas, si algún juego queda sin términos tras el filtrado TF-IDF o si algún bloque tiene varianza cero.

La fecha de referencia predeterminada es `2026-11-21` (`--reference-date AAAA-MM-DD`) y la semilla predeterminada para la descomposición SVD es `20261109` (`--seed`), garantizando la reproducibilidad de la antigüedad y de las proyecciones aleatorizadas. Los vocabularios, modelos SVD y parámetros de escalado se ajustan exclusivamente sobre los juegos que superan el filtro de reseñas mínimas.

## Uso

Después de fijar `data/interim/final_sample.csv` y contar con `data/raw/reviews.jsonl`:

```
.venv/bin/python scripts/modelo_B/prepare_features.py

```

Permite cambiar las rutas con `--sample`, `--raw-dir`, `--reviews` y `--output-dir`. También admite ajustar `--reference-date AAAA-MM-DD` (por defecto, `2026-11-21`), `--min-reviews` (por defecto, `10`), `--svd-components` (por defecto, `50`), `--seed` (por defecto, `20261109`) y los pesos de cada bloque mediante `--weight-metadata`, `--weight-reviews-text`, `--weight-description-text` y `--weight-review-stats`. Los archivos de salida se sobrescriben al volver a ejecutarlo; la muestra y los JSONL originales no se modifican.

## Resultado de la preparación

Ejecutado sobre la muestra final: **1.942 juegos y 187 variables** (85 de metadatos, 50 SVD de reseñas, 50 SVD de descripción y 2 estadísticas de reseña). Se usan 169.570 reseñas únicas tras eliminar 209 duplicados exactos. La SVD explica el 15,5 % de la varianza de las reseñas y el 10,1 % de las descripciones; cada bloque se escala a la misma varianza total que los metadatos (7,63) y el bloque de estadísticas lleva peso 0,5.

El TF-IDF de reseñas se calcula a nivel de juego para evitar que el vocabulario genérico de sentimiento domine el bloque. Con la media de vectores por reseña, 740 términos ubicuos concentraban el 45,7 % del peso; al agregar las reseñas en un documento por juego, `max_df=0,5` elimina esos términos y el vocabulario resultante es temático (`puzzles`, `boss`, `weapons`, `horror`, `strategy`, `multiplayer`).

## Script `cluster_metadata_text.py`

Entrena y compara K-Means y Ward sobre las features preparadas, con el mismo protocolo que el modelo A para que los resultados sean comparables:

1. Reutiliza las funciones genéricas de `scripts/modelo_A/cluster_metadata.py` (`read_inputs`, `validate_matrix`, `fit_model`, `select_best`, `relabel_by_size`, `top_rates` y `mean_column`).
2. Carga `features.csv` y `preprocessing.json`, además de `tfidf_reviews.npz`, `tfidf_description.npz` y `text_vocabulary.json` para perfilar los grupos con sus términos.
3. Prueba K-Means y Ward con `k=2..10`, calcula silhouette y Davies-Bouldin sobre la partición completa y mide la estabilidad de K-Means como ARI medio entre las 10 parejas de 5 semillas.
4. Selecciona con la misma regla que el A: mayor silhouette; empate, menor Davies-Bouldin; empate, menor `k`; empate, K-Means antes que Ward.
5. Reordena los grupos por tamaño y guarda `clustering_metrics.csv`, `cluster_assignments.csv`, `cluster_profiles.csv` y `clustering.json`. Solo persiste `modelo_B.pkl` si gana K-Means; si gana Ward, elimina cualquier `.pkl` anterior porque `AgglomerativeClustering` no ofrece `predict()`.

Los perfiles se expresan en unidades originales (precio en EUR, antigüedad en años, porcentaje de reseñas positivas y palabras por reseña) e incluyen los géneros, categorías y los 10 términos TF-IDF con mayor peso medio de cada grupo, tanto de reseñas como de descripciones.

```
.venv/bin/python scripts/modelo_B/cluster_metadata_text.py
```

Admite `--features`, `--preprocessing` y `--output-dir`, además de `--tfidf-reviews`, `--tfidf-description` y `--vocabulary` (por defecto dentro de `--output-dir`), `--k-min`, `--k-max` y las cinco semillas (`--seeds`).

## Resultado del clustering

Seleccionado **K-Means con k=2**: silhouette 0,0542, Davies-Bouldin 4,095 y ARI medio 0,963 entre semillas. Ward obtiene peores métricas en todos los `k` (silhouette máxima 0,0392 con k=2), así que se guarda `modelo_B.pkl`.

| Grupo | n | Precio | Edad | % positivas | Palabras/reseña | Términos de reseñas |
|---|---:|---:|---:|---:|---:|---|
| 0 — acción/combate | 1.043 (53,7 %) | 19,3 € | 6,7 años | 80,2 % | 162 | `boss`, `weapons`, `attack`, `damage`, `battle` |
| 1 — puzles/horror/novela | 899 (46,3 %) | 7,5 € | 5,1 años | 81,4 % | 85 | `puzzles`, `scenes`, `horror`, `relaxing`, `novel` |

El grupo 0 se asocia a Indie/Action/Adventure/RPG y el grupo 1 a Indie/Casual/Simulation. La separación es débil (silhouette 0,054), coherente con el análisis de features: metadatos solos alcanzan 0,1155 y añadir texto la reduce. La estructura débil es un resultado real, no un defecto de la representación.
