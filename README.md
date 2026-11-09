# IAO — Trabajo Final

Piloto de viabilidad para agrupar juegos de Steam mediante características
estructurales y de texto.

**Grupo 2** · Práctica Final IAO (2026/27) · Fase 0

## Estado actual

- Listado oficial de **199.314 juegos** obtenido con `IStoreService/GetAppList`.
- **505** juegos con `appdetails` descargados.
- **51 aptos** tras aplicar los filtros del piloto (10,1 %).
- Objetivo: **270 juegos**, 30 por cada uno de los 9 estratos.

La descarga sigue en curso por tandas hasta llenar las cuotas.

## Requisitos

- Python 3.14 (probado con 3.14.9).
- Dependencias del entorno virtual:

```bash
.venv/bin/pip install requests beautifulsoup4 lxml python-dateutil python-dotenv
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
data/
  steam_appids.csv        # 199.314 appids
  raw/                    # respuestas de Steam sin tocar (JSONL)
  interim/                # candidatos aptos con estrato
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
faltan para llegar a 30.

## Decisiones del piloto

- Universo: `type = game`, de pago (precio original > 0), lanzado, sin Early
  Access y con metadata mínima.
- Mínimo **100 reseñas totales** (todas las lenguas y tipos de compra).
- Mercado España: `cc=es`, precios en EUR.
- Estratos: 3 bandas de precio (`<10`, `10–24,99`, `≥25` €) × 3 bandas de
  edad (`<5`, `5–<10`, `≥10` años) = 9 estratos × 30 = **270 juegos**.
- Semilla de muestreo: `20261109`.
- Metacritic y propietarios (SteamSpy) quedan fuera de la selección y del
  clustering; solo se usarán como indicadores externos de validación.
- Sin muestra de contraste de 50–99 reseñas: `appdetails` no publica el
  recuento por debajo de ~100.

## Capas de datos

| Capa | Formato | Uso |
|---|---|---|
| Raw | JSONL | Copia fiel de Steam. Nunca se modifica. |
| Interim | CSV | Candidatos aptos con estrato asignado. |
| Processed | CSV | Resúmenes para inspección y análisis. |

El JSONL es la copia de seguridad cara de obtener; los CSV se pueden
regenerar en segundos sin volver a llamar a Steam.

## Pendiente

- Completar los ~14.500 candidatos que faltan para llenar las cuotas.
- Seleccionar los 270 juegos definitivos.
- Extraer reseñas en inglés para el conjunto de comparación.
- Preparar el A/B de modelos (metadatos vs. metadatos + texto).
