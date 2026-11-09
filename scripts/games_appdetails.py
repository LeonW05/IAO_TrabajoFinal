import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20261109
PAUSE = 1.5

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--shard", default="1/1", help="Shard actual/total, ej. 1/3")
args = parser.parse_args()

shard_index, shards = (int(x) for x in args.shard.split("/"))
if not 1 <= shard_index <= shards:
    raise SystemExit("Shard inválido: debe ser 1/3, 2/3, ...")

with (PROJECT_ROOT / "data" / "steam_appids.csv").open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

random.seed(SEED)
random.shuffle(rows)
shard_rows = rows[shard_index - 1 :: shards]

raw_dir = PROJECT_ROOT / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

done = set()
for path in raw_dir.glob("appdetails_candidates*.jsonl"):
    for line in path.open(encoding="utf-8"):
        try:
            item = json.loads(line)
            if item.get("status") == 200 and item.get("payload"):
                done.add(item["appid"])
        except json.JSONDecodeError:
            continue

pending = [row for row in shard_rows if row["appid"] not in done]
sample = pending[: args.limit]

tag = f"_shard{shard_index}of{shards}" if shards > 1 else ""
out_path = raw_dir / f"appdetails_candidates{tag}.jsonl"

print(f"shard {shard_index}/{shards} | ya hechos: {len(done)} | pendientes: {len(pending)} | esta tanda: {len(sample)}")
print(f"salida: {out_path}")

url = "https://store.steampowered.com/api/appdetails"

with out_path.open("a", encoding="utf-8") as out:
    for i, row in enumerate(sample, start=1):
        appid = row["appid"]

        r = None
        payload = {}
        status_code = 0

        for attempt in range(3):
            try:
                r = requests.get(url, params={"appids": appid, "cc": "es", "l": "english"}, timeout=30)
                status_code = r.status_code

                if status_code == 200:
                    break
                elif status_code == 429:
                    print(f"[{appid}] Rate limit (429). Esperando 30s...")
                    time.sleep(30)
                else:
                    time.sleep(10)
            except requests.RequestException as e:
                print(f"[{appid}] Error de red (intento {attempt+1}/3): {e}")
                time.sleep(10)

        if r is not None and status_code == 200:
            try:
                data = r.json()
                payload = data.get(str(appid), {})
            except ValueError:
                print(f"[{appid}] Error: La respuesta no es un JSON válido.")
                status_code = 999
        else:
            print(f"[{appid}] Falló después de 3 intentos. Status: {status_code}")

        out.write(json.dumps({
            "appid": appid,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status_code,
            "payload": payload,
        }, ensure_ascii=False) + "\n")
        out.flush()

        if i % 25 == 0 or i == len(sample):
            print(f"{i}/{len(sample)}")

        if status_code != 429:
            time.sleep(PAUSE)
