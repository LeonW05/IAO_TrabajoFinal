import csv
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "data" / "processed" / "appdetails_summary.csv"
ELIGIBLE = PROJECT_ROOT / "data" / "interim" / "eligible_candidates.csv"
TODAY = date.today().isoformat()

PRICE_BANDS = ["<10", "10-24.99", ">=25"]
AGE_BANDS = ["<5", "5-<10", ">=10"]


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def price_band(price):
    if price < 10:
        return "<10"
    if price < 25:
        return "10-24.99"
    return ">=25"


def age_band(age):
    if age < 5:
        return "<5"
    if age < 10:
        return "5-<10"
    return ">=10"


with SUMMARY.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

steps = [
    ("con datos", lambda r: True),
    ("type = game", lambda r: r["type"] == "game"),
    ("pago y precio > 0", lambda r: r["is_free"] == "False" and (number(r["price_eur"]) or 0) > 0),
    ("lanzado (fecha <= hoy)", lambda r: r["release_date"] and r["release_date"] <= TODAY),
    ("no Early Access", lambda r: r["early_access"] == "False"),
    ("metadata mínima", lambda r: bool(r["name"]) and bool(r["genres"]) and r["has_description"] == "True"),
    ("reseñas >= 100", lambda r: (number(r["reviews"]) or 0) >= 100),
]

current = rows
print(f"{'filtro':<28} {'quedan':>6} {'descartados':>11}")
for label, predicate in steps:
    before = len(current)
    current = [r for r in current if predicate(r)]
    print(f"{label:<28} {len(current):>6} {before - len(current):>11}")

eligible_ratio = len(current) / len(rows) if rows else 0
print(f"\naptos totales: {len(current)} de {len(rows)} ({eligible_ratio:.1%})")
print("\nDataset final (muestra aleatoria, sin cuotas por estrato):")
for target in (1500, 2000):
    missing = max(0, target - len(current))
    print(f"  objetivo {target}: faltan {missing} elegibles")

counts = Counter()
for r in current:
    counts[(price_band(number(r["price_eur"])), age_band(number(r["age_years"])))] += 1

print("\nEstratos (aptos):")
print(f"{'precio':<10}" + "".join(f"{band:>10}" for band in AGE_BANDS) + f"{'total':>10}")
for price in PRICE_BANDS:
    values = [counts[(price, age)] for age in AGE_BANDS]
    print(f"{price:<10}" + "".join(f"{v:>10}" for v in values) + f"{sum(values):>10}")

print("\nCobertura del piloto (objetivo 30 por estrato; solo informativo):")
for price in PRICE_BANDS:
    for age in AGE_BANDS:
        count = counts[(price, age)]
        print(f"  {price:<9} x {age:<6}: {count:>3}  faltan {max(0, 30 - count):>3}")

buckets = Counter()
for r in rows:
    reviews = number(r["reviews"])
    if reviews is None:
        buckets["sin dato"] += 1
    elif reviews < 50:
        buckets["<50"] += 1
    elif reviews < 100:
        buckets["50-99"] += 1
    else:
        buckets["100+"] += 1

print("\nDistribución de reseñas (todos):")
for bucket in ["sin dato", "<50", "50-99", "100+"]:
    print(f"  {bucket:<8} {buckets[bucket]:>4}")

for r in current:
    r["price_band"] = price_band(number(r["price_eur"]))
    r["age_band"] = age_band(number(r["age_years"]))
    r["stratum"] = f"{r['price_band']}_{r['age_band']}"

ELIGIBLE.parent.mkdir(parents=True, exist_ok=True)
with ELIGIBLE.open("w", encoding="utf-8", newline="") as f:
    output_fields = list(dict.fromkeys(fieldnames + ["price_band", "age_band", "stratum"]))
    writer = csv.DictWriter(f, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(current)

print(f"\naptos guardados en: {ELIGIBLE}")
