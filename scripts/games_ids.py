import csv
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
key = os.getenv("STEAM_API_KEY")
if not key:
    raise SystemExit("Falta la key en .env")

url = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
params = {
    "key": key,
    "include_dlc": "false",
    "include_software": "false",
    "include_videos": "false",
    "include_hardware": "false",
    "max_results": 50000
}

apps = []
page = 0

while True:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["response"]

    batch = data["apps"]
    apps.extend(batch)
    page += 1
    print(f"página {page}: {len(batch)} apps (total {len(apps)})")

    if not data.get("have_more_results"):
        break
    params["last_appid"] = data["last_appid"]

unique = {app["appid"]: app["name"] for app in apps}
rows = sorted(unique.items())

out = PROJECT_ROOT / "data" / "steam_appids.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["appid", "name"])
    writer.writerows(rows)

print(f"apps: {len(apps)}")
print(f"únicos: {len(rows)}")
print(f"CSV: {out}")
