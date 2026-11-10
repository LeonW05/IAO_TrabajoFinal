# Modelo A — Metadatos

Clustering de juegos usando solo metadatos de Steam. Metacritic y SteamSpy
se reservan para la validación externa.

## Seguimiento

- [x] Crear la carpeta y la documentación.
- [x] Definir las variables iniciales y su tratamiento.
- [x] Crear `prepare_features.py`: extraer, codificar y escalar los metadatos
  de la muestra final.
- [ ] Crear `cluster_metadata.py`: probar algoritmos y números de grupos.
- [ ] Evaluar silhouette, Davies-Bouldin y estabilidad entre semillas.
- [ ] Guardar y describir los grupos para compararlos con el modelo B.

## Variables iniciales

| Variable | Origen y tratamiento |
|---|---|
| Precio | `price_overview.initial` de los JSONL: precio original, sin descuento. Se exige EUR y valor positivo; se divide entre 100, se aplica `log1p` y se estandariza. |
| Antigüedad | `release_date` de la muestra: días entre el lanzamiento y la fecha de referencia, divididos entre 365,25. Se calcula sin redondear y se estandariza. |
| Géneros | Una columna `genres_<id>` por género presente en la muestra: 1 si el juego lo tiene, 0 si no. Un juego puede tener varios. |
| Categorías | Una columna `categories_<id>` por característica presente en la muestra, como multijugador o logros: 1 si la tiene, 0 si no. |
| Plataformas | Tres columnas 0/1: `platform_windows`, `platform_mac` y `platform_linux`. |

Estandarizar significa restar la media y dividir entre la desviación típica
poblacional de la muestra. En el precio se hace después de `log1p`, que reduce
las diferencias extremas. Si la desviación es cero, se usa un divisor de 1 y
la columna queda a cero. Las variables binarias no se escalan.

`appid` se conserva para identificar cada juego, pero no debe entrar en el
clustering. Tampoco se incluyen nombres, número de reseñas, estratos,
Metacritic ni SteamSpy.

## Script `prepare_features.py`

Prepara la tabla numérica; todavía no entrena ni asigna grupos:

1. Lee `data/interim/final_sample.csv`, comprueba que no esté vacío ni tenga
   identificadores inválidos o duplicados y ordena los juegos por `appid`.
2. Busca sus metadatos en `data/raw/appdetails_candidates*.jsonl`. Si un
   identificador se repite, conserva el último registro leído, igual que el
   script de resumen.
3. Valida los datos y aplica las transformaciones de la tabla anterior.
4. Guarda los resultados en `data/processed/modelo_A/`:
   - `features.csv`: una fila por juego, con `appid` y las variables numéricas.
   - `preprocessing.json`: columnas, etiquetas de géneros y categorías,
     medias, desviaciones, fecha de referencia y reglas aplicadas.

Se detiene si faltan metadatos, precio válido, fecha, géneros o indicadores
de plataformas. No elimina juegos silenciosamente ni rellena esos datos.
Las categorías ausentes se interpretan como lista vacía, con todos sus
indicadores a cero.

La fecha de referencia fija es `2026-11-09`, para que la antigüedad no cambie
según el día de ejecución. Conserva las fechas aproximadas que ya contiene
el CSV de muestra. Los parámetros y el vocabulario de columnas se calculan
sobre la muestra completa.

## Uso

Después de fijar `data/interim/final_sample.csv`:

```bash
.venv/bin/python scripts/modelo_A/prepare_features.py
```

Permite cambiar las rutas con `--sample`, `--raw-dir` y `--output-dir`, y la
fecha con `--reference-date AAAA-MM-DD`. Los archivos de salida se sobrescriben
al volver a ejecutarlo; la muestra y los JSONL originales no se modifican.
