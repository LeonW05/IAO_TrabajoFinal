"""Prepara las variables numéricas del modelo A sin ejecutar clustering."""

import argparse
import csv
import json
import math
import statistics
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = PROJECT_ROOT / "data/interim/final_sample.csv"
DEFAULT_RAW = PROJECT_ROOT / "data/raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/modelo_A"
REFERENCE_DATE = date(2026, 11, 9)
PLATFORMS = ("windows", "mac", "linux")


def read_sample(path):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"appid", "release_date"}.issubset(reader.fieldnames or []):
            raise ValueError("La muestra debe contener appid y release_date.")
        rows = list(reader)
    ids = [row["appid"] for row in rows]
    if not rows or any(not value.isdecimal() for value in ids):
        raise ValueError("La muestra está vacía o contiene appids inválidos.")
    if len(set(ids)) != len(ids):
        raise ValueError("La muestra contiene appids duplicados.")
    return sorted(rows, key=lambda row: int(row["appid"]))


def read_metadata(raw_dir, appids):
    records = {}
    paths = sorted(raw_dir.glob("appdetails_candidates*.jsonl"))
    if not paths:
        raise ValueError(f"No hay archivos appdetails_candidates*.jsonl en {raw_dir}.")
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSON inválido en {path}:{line_number}") from error
                appid = str(record.get("appid", ""))
                if appid in appids:
                    # Mismo orden y política de duplicados que appdetails_summary.py.
                    records[appid] = (record.get("payload") or {}).get("data") or {}
    missing = sorted(appid for appid in appids if not records.get(appid))
    if missing:
        raise ValueError(f"Faltan metadatos para {len(missing)} juegos: {', '.join(missing[:10])}")
    return records


def tags(data, key):
    values = data.get(key) or []
    if not isinstance(values, list):
        raise ValueError(f"El campo {key} no es una lista.")
    result = {}
    for item in values:
        identifier = str(item.get("id", ""))
        if not identifier.isdecimal():
            raise ValueError(f"Identificador inválido en {key}.")
        result[identifier] = item.get("description", "")
    return result


def prepare(sample, metadata, reference_date):
    games = []
    vocabularies = {"genres": {}, "categories": {}}
    for row in sample:
        appid = row["appid"]
        data = metadata[appid]
        try:
            price = data.get("price_overview") or {}
            initial = price.get("initial")
            if price.get("currency") != "EUR" or type(initial) is not int or initial <= 0:
                raise ValueError("Se requiere un precio original positivo en EUR.")
            released = date.fromisoformat(row["release_date"])
            if released > reference_date:
                raise ValueError("El lanzamiento es posterior a la fecha de referencia.")
            genres = tags(data, "genres")
            categories = tags(data, "categories")
            platforms = data.get("platforms")
            if not genres:
                raise ValueError("Faltan géneros.")
            if not isinstance(platforms, dict) or any(
                type(platforms.get(platform)) is not bool for platform in PLATFORMS
            ):
                raise ValueError("Faltan indicadores booleanos de plataformas.")
        except (ValueError, TypeError) as error:
            raise ValueError(f"appid {appid}: {error}") from error
        vocabularies["genres"].update(genres)
        vocabularies["categories"].update(categories)
        games.append({
            "appid": appid,
            "price_log1p": math.log1p(initial / 100),
            "age_years": (reference_date - released).days / 365.25,
            "genres": genres,
            "categories": categories,
            "platforms": platforms,
        })

    scaling = {}
    for feature in ("price_log1p", "age_years"):
        values = [game[feature] for game in games]
        std = statistics.pstdev(values)
        scaling[feature] = {"mean": statistics.mean(values), "std": std, "scale": std or 1.0}

    columns = ["price_log1p_scaled", "age_years_scaled"]
    columns += [f"{group}_{identifier}" for group, vocabulary in vocabularies.items()
                for identifier in sorted(vocabulary, key=int)]
    columns += [f"platform_{platform}" for platform in PLATFORMS]
    output = []
    for game in games:
        result = {"appid": game["appid"]}
        for feature, params in scaling.items():
            result[f"{feature}_scaled"] = (game[feature] - params["mean"]) / params["scale"]
        for group, vocabulary in vocabularies.items():
            for identifier in sorted(vocabulary, key=int):
                result[f"{group}_{identifier}"] = int(identifier in game[group])
        for platform in PLATFORMS:
            result[f"platform_{platform}"] = int(game["platforms"][platform])
        output.append(result)
    config = {
        "reference_date": reference_date.isoformat(),
        "rows": len(output),
        "feature_columns": columns,
        "identifier_column": "appid",
        "numeric_scaling": scaling,
        "price_transform": "log1p(initial / 100), EUR, sin descuento",
        "age_transform": "(reference_date - release_date).days / 365.25, sin redondear",
        "release_date_source": "CSV de muestra; conserva las aproximaciones del resumen",
        "binary_encoding": "0/1 sin escalado; vocabulario ajustado a la muestra",
        "vocabularies": vocabularies,
        "platforms": list(PLATFORMS),
        "missing_policy": "Error en precio, fecha, géneros o plataformas; categorías ausentes = lista vacía",
        "excluded": ["name", "metacritic", "reviews", "SteamSpy", "stratum"],
    }
    return output, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-date", type=date.fromisoformat, default=REFERENCE_DATE,
                        help="Fecha fija AAAA-MM-DD (por defecto: 2026-11-09).")
    args = parser.parse_args()
    try:
        sample = read_sample(args.sample)
        metadata = read_metadata(args.raw_dir, {row["appid"] for row in sample})
        rows, config = prepare(sample, metadata, args.reference_date)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    config["sample_path"] = str(args.sample.resolve())
    config["raw_dir"] = str(args.raw_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "features.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["appid"] + config["feature_columns"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Juegos: {len(rows)} | variables: {len(config['feature_columns'])}")
    print(f"Fecha de referencia: {args.reference_date}")
    print(f"Resultados: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
