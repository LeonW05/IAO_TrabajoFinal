# Modelo B — Metadatos y texto

Clustering de los mismos juegos del modelo A, añadiendo reseñas en inglés.
Metacritic y SteamSpy se reservan para la validación externa.

## Seguimiento

- [x] Crear la carpeta y la documentación.
- [x] Definir cómo seleccionar las reseñas y crear el script común
  `scripts/games_reviews.py`.
- [x] Descargar 100 reseñas en inglés por juego (2.000 registros válidos).
- [x] Auditar la cobertura y fijar el umbral de 10 reseñas (1.942 juegos).
- [x] Crear `prepare_features.py`: preparar la descripción y las reseñas,
  aplicar el umbral de 10 reseñas y combinarlas con los metadatos del
  modelo A.
- [ ] Crear `cluster_metadata_text.py`: probar algoritmos y números de
  grupos sobre la muestra de 1.942 juegos.
- [ ] Ejecutar el Modelo A sobre los 1.942 juegos (`A_sub`) para comparar
  sobre la misma muestra.

**Lo que sigue (en el README principal):** generación de figuras, comparación  
A × B (ARI y tabla cruzada), validaciones externas (SteamSpy, Metacritic) y  
redacción de conclusiones.

## Script `prepare_features.py`

Prepara la tabla numérica combinando metadatos y representaciones de texto; todavía no entrena ni asigna grupos:

1. Lee `data/interim/final_sample.csv` mediante `scripts/modelo_A/prepare_features.py`, comprueba que no esté vacío ni tenga identificadores inválidos o duplicados y ordena los juegos por `appid`.

   README.md
2. Lee `data/raw/reviews.jsonl` para los `appid` de la muestra, conserva el último registro válido (`status == 200` y `query_summary` presente), limpia el texto de las reseñas y excluye los juegos que no alcanzan el mínimo de 10 reseñas con texto (`--min-reviews`). Después construye **un documento por juego** concatenando sus reseñas únicas (elimina duplicados exactos) y ajusta el TF-IDF sobre esos documentos, no sobre las reseñas sueltas; así `max_df` filtra los términos que aparecen en más de la mitad de los juegos.
3. Busca los metadatos de los juegos restantes en `data/raw/appdetails_candidates*.jsonl`, valida los datos y aplica las transformaciones de los cuatro bloques junto con su reescalado por varianza.
4. Guarda los resultados en `data/processed/modelo_B/`:
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
