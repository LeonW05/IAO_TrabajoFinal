# Modelo A — Metadatos

Clustering de juegos usando solo metadatos de Steam. Metacritic y SteamSpy
se reservan para la validación externa.

## Seguimiento

- [x] Crear la carpeta y la documentación.
- [x] Definir las variables iniciales y su tratamiento.
- [x] Crear `prepare_features.py`: extraer, codificar y escalar los metadatos
  de la muestra final.
- [x] Crear `cluster_metadata.py`: probar algoritmos y números de grupos.
- [x] Evaluar silhouette, Davies-Bouldin y estabilidad entre semillas.
- [x] Guardar y describir los grupos para compararlos con el modelo B.

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

La fecha de referencia predeterminada es `2026-11-21`, necesaria para incluir
juegos de la muestra lanzados hasta ese día. Así, la antigüedad es reproducible
y no depende de la fecha en que se ejecute el script. Puede fijarse otra fecha
con `--reference-date AAAA-MM-DD`; se debe usar la misma al regenerar las
variables para reproducir un análisis. Se conservan las fechas aproximadas del
CSV de muestra, y los parámetros y el vocabulario se calculan sobre la muestra
completa.

## Uso

Después de fijar `data/interim/final_sample.csv`:

```bash
.venv/bin/python scripts/modelo_A/prepare_features.py
```

Permite cambiar las rutas con `--sample`, `--raw-dir` y `--output-dir`, y la
fecha con `--reference-date AAAA-MM-DD` (por defecto, `2026-11-21`). Los archivos
de salida se sobrescriben al volver a ejecutarlo; la muestra y los JSONL
originales no se modifican.

## Script `cluster_metadata.py`

Compara particiones; no vuelve a preparar variables ni toca la muestra:

1. Lee `features.csv` y `preprocessing.json`. Exige las mismas columnas, en el

   mismo orden, e ignora `appid` al ajustar los grupos.
2. Prueba k-means (`n_init=10`, Lloyd) y clustering aglomerativo (enlace de

   Ward) para k de 2 a 10. La distancia es la euclídea, coherente con el
    escalado de precio y antigüedad.
3. En cada combinación calcula silhouette y Davies-Bouldin sobre la partición
   completa. K-Means se ajusta con cinco semillas (`20261109` y las cuatro
   siguientes); se guarda el ARI medio entre las diez parejas de particiones.
   Las métricas y asignaciones de K-Means corresponden a la primera semilla.
   Ward no usa semilla, así que su estabilidad queda vacía.
4. Elige la combinación con mayor silhouette. En un empate gana el menor
   Davies-Bouldin y, después, el menor k. El ARI se informa, pero no interviene
   en la selección.
5. Numera los grupos de mayor a menor tamaño y guarda, en el mismo directorio:
   - `clustering_metrics.csv`: métricas de todas las combinaciones válidas.
   - `cluster_assignments.csv`: `appid` y grupo del modelo elegido.
   - `cluster_profiles.csv`: tamaño, peso, precio medio en EUR, antigüedad
   media en años, proporción de plataformas y géneros y categorías más
   frecuentes.
   - `clustering.json`: semillas, regla de selección y rutas.

Una combinación se descarta si algún grupo tiene menos de dos juegos, porque
silhouette y Davies-Bouldin no se pueden calcular.

Las dos casillas de evaluación y descripción se cierran al ejecutarlo sobre
`features.csv` de la muestra final. Volver a lanzarlo sobrescribe esas cuatro
salidas; no modifica `features.csv` ni `preprocessing.json`.

```bash
.venv/bin/python scripts/modelo_A/cluster_metadata.py
```

`--features`, `--preprocessing` y `--output-dir` cambian las rutas. `--k-min`,
`--k-max` cambian el rango de grupos. `--seeds` permite especificar exactamente
cinco semillas para K-Means; la primera produce las métricas y las asignaciones
que se guardan. El ARI medio se reporta en `clustering_metrics.csv` y no afecta
la selección del algoritmo ni de k.

## Resultado sobre la muestra final

Ejecutado sobre 2.000 juegos y 85 variables. Se eligió **Ward con k=2**
(silhouette 0,1527; Davies-Bouldin 2,6188).

| Grupo | Juegos | Peso | Precio medio | Antigüedad media |
|---|---:|---:|---:|---:|
| 0 | 1.578 | 78,9 % | 12,61 EUR | 6,96 años |
| 1 | 422 | 21,1 % | 17,71 EUR | 1,89 años |

El grupo 0 reúne juegos más baratos y antiguos; el grupo 1, juegos más caros
y recientes. La estabilidad de K-Means es alta para k=2 (ARI 1,000), aunque
Ward obtiene mejor silhouette. Ward no produce un modelo predictivo `.pkl`.
