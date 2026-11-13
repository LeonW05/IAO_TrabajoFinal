# IAO — Trabajo Final

Piloto de viabilidad para agrupar juegos de Steam mediante características
estructurales y de texto.

**Grupo 2** · Práctica Final IAO (2026/27) · Fase 0

## Estado actual

- Listado oficial de **199.314 juegos** obtenido con `IStoreService/GetAppList`.
- **2.505** juegos con `appdetails` descargados.
- **265 aptos** tras aplicar los filtros (10,6 %).
- **Piloto Fase 0:** 270 juegos (30 por estrato) para validar el pipeline.
- **Dataset final:** muestra aleatoria de ~2.000 elegibles; no se filtra por
Metacritic.

La descarga sigue en curso hasta tener suficientes elegibles para la muestra
final.

## Requisitos

- Python 3.14 (probado con 3.14.9).
- Dependencias del entorno virtual:

```bash
.venv/bin/pip install requests beautifulsoup4 lxml python-dateutil python-dotenv numpy scikit-learn
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
  modelo_A/               # futuros scripts de preparación y clustering con metadatos
  modelo_B/               # futuros scripts de clustering con metadatos y texto
data/
  steam_appids.csv        # 199.314 appids
  raw/                    # respuestas de Steam sin tocar (JSONL)
  interim/                # candidatos aptos y muestra final (CSV)
  processed/              # resúmenes CSV para análisis
docs/
  Propuesta_Tema_Practica_Final_IAO-1.pdf
  TODO_Fase0.md
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

La tabla final indica cuántos juegos aptos hay en cada estrato y cuántos
faltan para llegar a 30. Esa cuota es solo del piloto; el dataset final se
genera con la muestra aleatoria de la sección siguiente.

### 5. Elegir la muestra

Cuando haya al menos 2.000 juegos elegibles y se hayan regenerado el resumen
y los candidatos (pasos 3 y 4), ejecutar:

```bash
.venv/bin/python scripts/select_sample.py
```

El script selecciona 2.000 juegos al azar con semilla fija `20261109`, sin
cuotas por estrato ni filtro por Metacritic, y guarda la muestra en
`data/interim/final_sample.csv`. También muestra la distribución por estrato
y la cobertura de Metacritic.

Si se decide reducir el objetivo por falta de tiempo, se puede seleccionar
una muestra de 1.500 juegos usando `--size 1500`. El script se detiene si no
hay suficientes elegibles o si `final_sample.csv` ya existe. Usarlo al cerrar
la recopilación de candidatos, antes de descargar las reseñas; la muestra
elegida se mantiene fija para las fases posteriores.

## Decisiones del piloto

- Universo: `type = game`, de pago (precio original > 0), lanzado, sin Early
Access y con metadata mínima.
- Mínimo **100 reseñas totales** (todas las lenguas y tipos de compra).
- Mercado España: `cc=es`, precios en EUR.
- Estratos del piloto: 3 bandas de precio (`<10`, `10–24,99`, `≥25` €) × 3
bandas de edad (`<5`, `5–<10`, `≥10` años) = 9 estratos × 30 = **270
juegos**. El dataset final no usa cuotas fijas.
- Semilla de muestreo: `20261109`.
- Metacritic y propietarios (SteamSpy) quedan fuera de la selección y del
clustering; solo se usarán como indicadores externos de validación.
- Sin muestra de contraste de 50–99 reseñas: `appdetails` no publica el
recuento por debajo de ~100.

## Plan del dataset final

El piloto de 270 sirve para validar el pipeline. El dataset final se construye
con una **muestra aleatoria de elegibles**, no con cuotas fijas de 30 por
estrato. Reglas:

- Universo: los mismos filtros del piloto (`type = game`, no gratuito, precio
original > 0, lanzado, sin Early Access, metadata mínima y ≥100 reseñas).
- **No filtrar por Metacritic.** Se guarda como columna y se usa solo después
del clustering, junto con SteamSpy. Los juegos sin nota quedan fuera de las
comparaciones que la requieran, pero sí entran en el clustering.
- Muestra: 2.000 juegos elegibles (o 1.500 si el tiempo aprieta) con semilla
fija `20261109`. Los estratos de precio/edad se usan para describir
cobertura, no para forzar el tamaño de cada celda.
- Reseñas: inglés y un máximo de 30–50 por juego para controlar el coste.
- Clustering: modelo A (metadatos) y modelo B (metadatos + texto); silhouette,
Davies-Bouldin y estabilidad entre semillas.
- Análisis externo: Metacritic y SteamSpy solo después de fijar los grupos.
Reportar cobertura por cluster y repetir el análisis en el subconjunto con
Metacritic como comprobación de robustez.

### Muestra aleatoria de elegibles

1. Regenerar los candidatos aptos:

```bash
.venv/bin/python scripts/analyze_candidates.py
```

2. Comprobar que `data/interim/eligible_candidates.csv` tiene al menos el

   tamaño elegido (2.000; 1.500 si el tiempo aprieta). Si no, seguir
    descargando `appdetails` por tandas.
3. Generar la muestra con semilla fija y guardarla en

   `data/interim/final_sample.csv`:

```bash
.venv/bin/python scripts/select_sample.py

# Alternativa si no se llega a 2.000 elegibles:
.venv/bin/python scripts/select_sample.py --size 1500
```

El script utiliza la semilla fija `20261109`, informa de la cobertura por
estrato y de Metacritic, y se detiene si faltan elegibles o ya existe la
muestra. Ejecuta solo una de las dos opciones anteriores.

4. No modificar `final_sample.csv` después de fijarlo. Todas las fases

   posteriores (reseñas, clustering y análisis externo) usan esa muestra.

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

- Completar la descarga hasta tener el tamaño elegido (2.000; 1.500 si el
tiempo aprieta).
- Generar `data/interim/final_sample.csv` con la muestra aleatoria (semilla
`20261109`).
- Extraer reseñas en inglés (30–50 por juego).
- Preparar el A/B de modelos (metadatos vs. metadatos + texto).
- Cruzar los clusters con Metacritic y SteamSpy y reportar la cobertura.
