# IAO — Trabajo Final

Implementación para agrupar juegos de Steam mediante características
estructurales y de texto.

**Grupo 2** · Práctica Final IAO (2026/27)

## Estado actual

- Listado oficial de **199.314 juegos** obtenido con `IStoreService/GetAppList`.
- **19.236** juegos con `appdetails` descargados.
- **2.075 aptos** tras aplicar los filtros (10,8 %).
- **Dataset final:** 2.000 juegos elegibles seleccionados al azar con semilla
`20261109`; no se filtra por Metacritic.
- **Modelo A:** clustering con metadatos ejecutado sobre la muestra final
(Ward, k=2).
- **Modelo B:** features de metadatos y texto y clustering ejecutados sobre
1.942 juegos (K-Means, k=2).
- **Comparación A × B:** 1.942 juegos alineados, ARI `-0,0018`.
- **Validación externa:** SteamSpy descargado para los 2.000 appids (100 %)
  y cruzado junto con Metacritic (16,9 % de cobertura) con los clusters de A,
  `A_sub` y B en `modelos/validacion_externa/`.

La muestra final está fijada y los Modelos A y B ya están ejecutados. La
descarga de `appdetails` no continúa salvo que se quiera ampliar la muestra.

## Requisitos

- Python 3.14 (probado con 3.14.9).
- Dependencias del entorno virtual:

```bash
.venv/bin/pip install requests beautifulsoup4 lxml python-dateutil python-dotenv numpy scikit-learn joblib
```

Estas dependencias se encuentran en `requirements.txt`. Para instalar:

- Windows:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- macOS / Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Configuración

1. Copia la plantilla y rellena la clave:

```bash
cp .env.example .env
```

2. Consigue una Steam Web API key gratuita en

   <https://steamcommunity.com/dev/apikey> y pégala en `.env`:

```text
STEAM_API_KEY=tu_clave_aqui
```

`.env` está en `.gitignore`; `.env.example` sí se sube.

## Estructura

```text
scripts/
  games_ids.py            # lista de appids (IStoreService)
  games_appdetails.py     # descarga appdetails por tandas/shards
  appdetails_summary.py   # JSONL raw -> CSV legible
  analyze_candidates.py   # embudo de filtros y cobertura por estrato
  select_sample.py        # muestra aleatoria final de 2.000 o 1.500 juegos
  modelo_A/               # preparación y clustering con metadatos
  modelo_B/               # preparación de texto y clustering con metadatos + texto
  compare_models.py       # comparación de asignaciones A_sub × B (ARI y tabla cruzada)
  fetch_steamspy.py       # descarga de propietarios estimados (SteamSpy)
  external_validation.py  # validación externa (SteamSpy + Metacritic) por cluster
data/
  steam_appids.csv        # 199.314 appids
  raw/                    # respuestas de Steam sin tocar (JSONL)
  interim/                # candidatos aptos y muestra final (CSV)
  processed/              # resúmenes CSV para análisis
docs/
  Propuesta_Tema_Practica_Final_IAO-1.pdf
```

## Uso

### 1. Listado de appids

```bash
.venv/bin/python scripts/games_ids.py
```

Genera `data/steam_appids.csv`. Tarda unos segundos (4 peticiones).

### 2. Descargar appdetails

Por tandas:

```bash
.venv/bin/python scripts/games_appdetails.py --limit 2000
```

O repartido entre varias personas:

```bash
.venv/bin/python scripts/games_appdetails.py --limit 5000 --shard 1/3
.venv/bin/python scripts/games_appdetails.py --limit 5000 --shard 2/3
.venv/bin/python scripts/games_appdetails.py --limit 5000 --shard 3/3
```

Cada juego tarda ~1,5 s (límite de Steam). El script es reanudable: al
relanzarlo salta los appids ya descargados.

### 3. Generar el CSV resumen

```bash
.venv/bin/python scripts/appdetails_summary.py
```

### 4. Ver filtros y cobertura por estrato

```bash
.venv/bin/python scripts/analyze_candidates.py
```

La tabla muestra el número de juegos aptos por estrato como descripción de
la cobertura; no se aplican cuotas de selección por estrato.

### 5. Selección de muestra (si se reconstruye el dataset)

La muestra final ya está fijada en `data/interim/final_sample.csv`; no se debe
regenerar para reproducir los análisis actuales. Si en el futuro se reconstruye
el dataset desde cero, `scripts/select_sample.py` selecciona 2.000 juegos con
semilla `20261109`, sin cuotas por estrato ni filtro por Metacritic. El script
también admite `--size 1500` para un dataset nuevo de menor tamaño.
### 6. Validación externa (SteamSpy y Metacritic)

```bash
.venv/bin/python scripts/fetch_steamspy.py
.venv/bin/python scripts/external_validation.py

## Criterios del dataset

El dataset final se construyó con una **muestra aleatoria de elegibles**, sin
cuotas fijas por estrato. Criterios aplicados:

- Universo: `type = game`, de pago (precio original > 0), lanzado, sin Early
  Access, con metadata mínima y al menos 100 reseñas totales.
- Mercado España (`cc=es`), precios en EUR.
- Semilla de muestreo: `20261109`.
- Metacritic y propietarios (SteamSpy) quedan fuera de la selección y del
  clustering; se reservan para validaciones externas.
- La API de Steam no ofrece un recuento fiable para el intervalo de 50–99
  reseñas; por eso no se utilizó una muestra de contraste en ese rango.
- **No filtrar por Metacritic.** Se guarda como columna y se usa solo después
del clustering, junto con SteamSpy. Los juegos sin nota quedan fuera de las
comparaciones que la requieran, pero sí entran en el clustering.
- Muestra fijada: 2.000 juegos elegibles seleccionados con semilla `20261109`.
  Los estratos de precio/edad describen la cobertura, no determinan el tamaño
  de cada celda.
- Reseñas: se recopilaron hasta 100 reseñas en inglés por juego; el Modelo B
  incluye los juegos con al menos 10 reseñas utilizables.
- Clustering: modelo A (metadatos) y modelo B (metadatos + texto); silhouette,
Davies-Bouldin y estabilidad entre semillas.
- Análisis externo: Metacritic y SteamSpy solo después de fijar los grupos.
Reportar cobertura por cluster y repetir el análisis en el subconjunto con
Metacritic como comprobación de robustez.


## Capas de datos

| Capa | Formato | Uso |
|---|---|---|
| Raw | JSONL | Copia fiel de Steam. Nunca se modifica. |
| Interim | CSV | Candidatos aptos, estrato y muestra final. |
| Processed | CSV | Resúmenes para inspección y análisis. |

El JSONL es la copia de seguridad cara de obtener; los CSV se pueden
regenerar en segundos sin volver a llamar a Steam.

## Pendiente

El estado y las tareas específicas de cada modelo se mantienen por separado:

- [Modelo A: metadatos](scripts/modelo_A/README.md).
- [Modelo B: metadatos y texto](scripts/modelo_B/README.md).

Tareas generales del proyecto:

- [x] Completar la descarga hasta tener el tamaño elegido (2.000; 1.500 si el
  tiempo aprieta).
- [x] Generar `data/interim/final_sample.csv` con la muestra aleatoria (semilla
  `20261109`).
- [x] Extraer reseñas en inglés (100 por juego, 2.000 registros válidos).
- [x] Auditar la cobertura y fijar el umbral de 10 reseñas para el Modelo B
  (1.942 juegos).
- [x] Modelo A: clustering con metadatos (Ward, k=2).
- [x] **Modelo B: preparar features** — crear `prepare_features.py` para
  TF-IDF de reseñas (documento por juego) y descripción, % positivas, longitud
  media, SVD, y unir con metadatos.
- [x] **Modelo B: clustering** — crear `cluster_metadata_text.py` y ejecutar
  K-Means/Ward sobre la muestra de 1.942 juegos (K-Means, k=2; silhouette
  0,0542, DB 4,095, ARI 0,963) y guardar assignments.
- [x] **Modelo A_sub y comparación A × B** — `scripts/compare_models.py` filtra
  A a los appids de B, calcula el ARI y genera asignaciones alineadas y tabla
  cruzada. Resultado: 1.942 juegos comparados, ARI **-0,0018**; tabla y resumen
  en `modelos/comparacion_A_B/`.
- [x] **Validaciones externas** — descargar SteamSpy para 2.000 appids,
  cruzar con clusters y reportar cobertura de Metacritic por cluster.
- [ ] Generar figuras para la memoria (heatmap de perfiles, precio/edad por
  cluster, términos TF-IDF por cluster, validación externa).
- [ ] Documentar limitaciones (sesgo de idioma no anglófono) y conclusiones.
