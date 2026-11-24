"""Descarga datos de SteamSpy (appdetails) para la muestra final.

Lee los appids de data/interim/final_sample.csv y guarda las respuestas
en data/raw/steamspy*.jsonl. Es reanudable y admite reparto por shards.
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "interim" / "final_sample.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
STEAMSPY_URL = "https://steamspy.com/api.php"
DEFAULT_PAUSE = 1.2


def read_sample_appids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "appid" not in (reader.fieldnames or []):
            raise ValueError(f"{path} no contiene la columna 'appid'.")
        appids = [row["appid"].strip() for row in reader if row.get("appid")]
    if not appids or any(not a.isdecimal() for a in appids):
        raise ValueError("La muestra está vacía o contiene appids inválidos.")
    return sorted(set(appids), key=int)


def load_done(raw_dir: Path) -> set[str]:
    done = set()
    for path in raw_dir.glob("steamspy*.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if item.get("status") == 200 and isinstance(item.get("payload"), dict):
                        done.add(str(item["appid"]))
                except json.JSONDecodeError:
                    continue
    return done


def fetch_one(session: requests.Session, appid: str) -> tuple[int, dict]:
    status_code = 0
    payload = {}
    for attempt in range(3):
        try:
            r = session.get(
                STEAMSPY_URL,
                params={"request": "appdetails", "appid": appid},
                timeout=30,
            )
            status_code = r.status_code
            if status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, dict):
                        return status_code, data
                    status_code = 999
                except ValueError:
                    print(f"[{appid}] Respuesta no es JSON válido.")
                    status_code = 999
                break
            elif status_code == 429:
                print(f"[{appid}] Rate limit (429). Esperando 30s...")
                time.sleep(30)
            else:
                print(f"[{appid}] HTTP {status_code} (intento {attempt + 1}/3).")
                time.sleep(10)
        except requests.RequestException as e:
            print(f"[{appid}] Error de red (intento {attempt + 1}/3): {e}")
            time.sleep(10)
    return status_code, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Máximo de juegos en esta tanda.")
    parser.add_argument("--shard", default="1/1", help="Shard actual/total, ej. 1/4.")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE, help="Segundos entre peticiones.")
    args = parser.parse_args()

    try:
        shard_index, shards = (int(x) for x in args.shard.split("/"))
    except ValueError:
        parser.error("Shard inválido: debe tener formato actual/total, ej. 1/4.")
    if not 1 <= shard_index <= shards:
        parser.error("Shard inválido: debe cumplir 1 <= actual <= total.")

    appids = read_sample_appids(args.sample)
    shard_appids = appids[shard_index - 1 :: shards]

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(args.raw_dir)
    pending = [appid for appid in shard_appids if appid not in done]
    batch = pending[: args.limit] if args.limit is not None else pending

    tag = f"_shard{shard_index}of{shards}" if shards > 1 else ""
    out_path = args.raw_dir / f"steamspy{tag}.jsonl"

    print(
        f"shard {shard_index}/{shards} | total shard: {len(shard_appids)} | "
        f"ya hechos: {len(shard_appids) - len(pending)} | pendientes: {len(pending)} | "
        f"esta tanda: {len(batch)}"
    )
    print(f"salida: {out_path}")

    if not batch:
        print("Nada pendiente en este shard.")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": "IAO-UC3M-Grupo2-AcademicProject/1.0"})

    with out_path.open("a", encoding="utf-8") as out:
        for i, appid in enumerate(batch, start=1):
            status_code, payload = fetch_one(session, appid)
            out.write(
                json.dumps(
                    {
                        "appid": appid,
                        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "status": status_code,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

            if i % 25 == 0 or i == len(batch):
                print(f"{i}/{len(batch)}")

            if i < len(batch) and status_code != 429:
                time.sleep(args.pause)


if __name__ == "__main__":
    main()