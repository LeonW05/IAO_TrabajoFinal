import csv
import json
import re
from datetime import date, datetime
from pathlib import Path

from dateutil import parser as date_parser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed" / "appdetails_summary.csv"
TODAY = date.today()


def parse_release(raw):
    if not raw:
        return None
    text = raw.strip()
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 7, 1)
    if re.fullmatch(r"[A-Za-z]+ \d{4}", text):
        try:
            return date_parser.parse(text, default=datetime(2000, 1, 15)).date()
        except ValueError:
            return None
    try:
        return date_parser.parse(text).date()
    except (ValueError, OverflowError):
        return None


by_id = {}
for path in sorted(RAW_DIR.glob("appdetails_candidates*.jsonl")):
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            by_id[record["appid"]] = record

rows = []
for appid, record in by_id.items():
    d = (record.get("payload") or {}).get("data") or {}
    if not d:
        continue

    price = d.get("price_overview") or {}
    initial = price.get("initial")
    release_raw = (d.get("release_date") or {}).get("date")
    released = parse_release(release_raw)
    age = round((TODAY - released).days / 365.25, 1) if released else None

    genres = [g.get("description", "") for g in d.get("genres", [])]
    categories = [c.get("description", "") for c in d.get("categories", [])]
    early_access = any("early access" in text.lower() for text in genres + categories)

    rows.append({
        "appid": appid,
        "name": d.get("name"),
        "type": d.get("type"),
        "is_free": d.get("is_free"),
        "price_eur": (initial / 100) if isinstance(initial, int) else None,
        "currency": price.get("currency"),
        "release_raw": release_raw,
        "release_date": released.isoformat() if released else None,
        "age_years": age,
        "genres": "|".join(genres),
        "early_access": early_access,
        "has_description": bool(d.get("detailed_description")),
        "metacritic": (d.get("metacritic") or {}).get("score"),
        "reviews": (d.get("recommendations") or {}).get("total"),
    })

rows.sort(key=lambda r: int(r["appid"]))
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"juegos: {len(rows)}")
print(f"CSV: {OUT}\n")

print(f"{'appid':<9} {'nombre':<38} {'precio':>7} {'fecha':<11} {'edad':>5} {'EA':<5} {'reseñas':>8}")
for row in rows[:12]:
    name = (row["name"] or "")[:38]
    print(
        f"{row['appid']:<9} {name:<38} {str(row['price_eur'] or ''):>7} "
        f"{str(row['release_date'] or ''):<11} {str(row['age_years'] or ''):>5} "
        f"{str(row['early_access']):<5} {str(row['reviews'] or ''):>8}"
    )
