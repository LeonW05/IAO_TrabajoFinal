"""Selecciona la muestra aleatoria final a partir de los juegos elegibles."""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data" / "interim" / "eligible_candidates.csv"
OUTPUT = PROJECT_ROOT / "data" / "interim" / "final_sample.csv"
SEED = 20261109


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, choices=(1500, 2000), default=2000,
        help="Número de juegos de la muestra (por defecto: 2000).",
    )
    args = parser.parse_args()

    if OUTPUT.exists():
        raise SystemExit(
            f"La muestra ya existe: {OUTPUT}. No se sobrescribe una muestra fijada."
        )
    if not SOURCE.is_file():
        raise SystemExit(
            f"No existe {SOURCE}. Ejecuta primero scripts/analyze_candidates.py."
        )

    with SOURCE.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        required = {"appid", "stratum", "metacritic"}
        if not required.issubset(fieldnames or []):
            raise SystemExit("El CSV de elegibles debe incluir appid, stratum y metacritic.")
        rows = list(reader)

    appids = [row["appid"] for row in rows]
    if any(not appid or not appid.isdecimal() for appid in appids):
        raise SystemExit("El CSV contiene identificadores appid vacíos o inválidos.")
    if len(set(appids)) != len(appids):
        raise SystemExit("El CSV contiene appids duplicados; regenera los candidatos.")
    if len(rows) < args.size:
        raise SystemExit(
            f"Solo hay {len(rows)} elegibles; faltan {args.size - len(rows)} "
            f"para seleccionar {args.size}. Continúa descargando y regenera los candidatos."
        )

    # Orden estable para que reorganizar el CSV no altere la selección.
    rows.sort(key=lambda row: int(row["appid"]))
    sample = random.Random(SEED).sample(rows, args.size)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with OUTPUT.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample)
    except FileExistsError:
        raise SystemExit(f"La muestra ya existe: {OUTPUT}. No se ha sobrescrito.")

    metacritic_count = sum(bool(row["metacritic"].strip()) for row in sample)
    print(f"Elegibles disponibles: {len(rows)} | muestra: {len(sample)} | semilla: {SEED}")
    print(f"Con Metacritic: {metacritic_count} ({metacritic_count / len(sample):.1%})")
    print("Por estrato (sin cuotas):")
    for stratum, count in sorted(Counter(row["stratum"] for row in sample).items()):
        print(f"  {stratum}: {count}")
    print(f"Muestra guardada en: {OUTPUT}")


if __name__ == "__main__":
    main()
