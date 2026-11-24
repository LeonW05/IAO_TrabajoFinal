"""Validación externa de los modelos A, A_sub y B con Metacritic y SteamSpy.

Cruza las asignaciones de clustering con las puntuaciones de Metacritic
(de data/interim/final_sample.csv) y los rangos estimados de propietarios
de SteamSpy (de data/raw/steamspy*.jsonl). Reporta cobertura de Metacritic
por cluster, distribuciones por tramo, contrastes no paramétricos y una
comprobación de robustez restringida al subconjunto con nota de Metacritic.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "interim" / "final_sample.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MODEL_A = PROJECT_ROOT / "modelos" / "modelo_A" / "cluster_assignments.csv"
DEFAULT_MODEL_A_SUB = PROJECT_ROOT / "modelos" / "comparacion_A_B" / "A_sub_assignments.csv"
DEFAULT_MODEL_B = PROJECT_ROOT / "modelos" / "modelo_B" / "cluster_assignments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "modelos" / "validacion_externa"

OWNERS_RE = re.compile(r"^\s*([\d,]+)\s*\.\.\s*([\d,]+)\s*$")


def read_sample_metacritic(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"appid", "metacritic"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} debe contener las columnas 'appid' y 'metacritic'.")
        records = {}
        for row in reader:
            appid = row["appid"].strip()
            raw_mc = (row.get("metacritic") or "").strip()
            mc = None
            if raw_mc:
                try:
                    val = float(raw_mc)
                    if np.isfinite(val) and val > 0:
                        mc = val
                except ValueError:
                    mc = None
            records[appid] = {
                "name": row.get("name", ""),
                "metacritic": mc,
            }
    if not records:
        raise ValueError(f"{path} está vacío.")
    return records


def parse_owners(raw: str | None) -> tuple[int, int, str] | None:
    if not isinstance(raw, str):
        return None
    match = OWNERS_RE.match(raw)
    if not match:
        return None
    low = int(match.group(1).replace(",", ""))
    high = int(match.group(2).replace(",", ""))
    if high <= 0 or high < low:
        return None
    label = f"{low:,} .. {high:,}"
    return low, high, label


def read_steamspy(raw_dir: Path, appids: set[str]) -> dict[str, dict]:
    paths = sorted(raw_dir.glob("steamspy*.jsonl"))
    if not paths:
        raise ValueError(f"No hay archivos steamspy*.jsonl en {raw_dir}.")
    records = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSON inválido en {path}:{line_number}") from error
                appid = str(item.get("appid", ""))
                if appid not in appids:
                    continue
                payload = item.get("payload")
                if item.get("status") == 200 and isinstance(payload, dict):
                    parsed = parse_owners(payload.get("owners"))
                    records[appid] = {
                        "owners_raw": payload.get("owners", ""),
                        "owners_parsed": parsed,
                        "average_forever": payload.get("average_forever"),
                        "median_forever": payload.get("median_forever"),
                    }
    return records


def read_assignments(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["appid", "cluster"]:
            raise ValueError(f"{path}: se esperan las columnas appid,cluster.")
        return {row["appid"].strip(): int(row["cluster"]) for row in reader}


def compare_groups_numeric(groups: list[np.ndarray]) -> dict:
    """Contraste no paramétrico entre clusters (Mann-Whitney U para k=2, Kruskal para k>2)."""
    non_empty = [g for g in groups if len(g) > 0]
    if len(non_empty) < 2:
        return {"test": None, "statistic": None, "p_value": None, "effect_size": None}
    if len(groups) == 2:
        g0, g1 = groups
        if len(g0) == 0 or len(g1) == 0:
            return {"test": "mannwhitneyu", "statistic": None, "p_value": None, "effect_size": None}
        if np.all(g0 == g0[0]) and np.all(g1 == g0[0]):
            return {"test": "mannwhitneyu", "statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
        res = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        # Correlación rank-biserial con signo: > 0 indica cluster 0 > cluster 1
        r_rb = float((2.0 * res.statistic) / (len(g0) * len(g1)) - 1.0)
        return {
            "test": "mannwhitneyu",
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "rank_biserial_r": r_rb,
            "direction_note": "r > 0 implica valores mayores en cluster 0 que en cluster 1",
        }
    try:
        res = stats.kruskal(*non_empty)
        return {
            "test": "kruskal",
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
        }
    except ValueError:
        return {"test": "kruskal", "statistic": 0.0, "p_value": 1.0}


def chi2_from_table(table: np.ndarray) -> dict:
    # Elimina columnas completamente vacías para evitar división por cero
    valid_cols = table.sum(axis=0) > 0
    valid_rows = table.sum(axis=1) > 0
    clean = table[np.ix_(valid_rows, valid_cols)]
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return {"test": "chi2", "statistic": None, "p_value": None, "dof": None}
    chi2, p_val, dof, _ = stats.chi2_contingency(clean)
    return {
        "test": "chi2",
        "statistic": float(chi2),
        "p_value": float(p_val),
        "dof": int(dof),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, decimals: int = 4) -> str:
    return "" if value is None else f"{value:.{decimals}f}"


def evaluate_model(
    model_name: str,
    assignments: dict[str, int],
    sample_data: dict[str, dict],
    steamspy_data: dict[str, dict],
    tier_order: list[tuple[int, int, str]],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    tier_to_rank = {label: idx for idx, (_, _, label) in enumerate(tier_order)}
    rank_to_label = {idx: label for idx, (_, _, label) in enumerate(tier_order)}
    clusters = sorted(set(assignments.values()))

    # --- 1. Metacritic ---
    mc_rows = []
    mc_groups = []
    mc_cov_table = []
    for c in clusters:
        c_ids = [aid for aid, cl in assignments.items() if cl == c]
        mc_vals = np.array(
            [sample_data[aid]["metacritic"] for aid in c_ids if sample_data[aid]["metacritic"] is not None],
            dtype=float,
        )
        mc_groups.append(mc_vals)
        n_total = len(c_ids)
        n_mc = len(mc_vals)
        mc_cov_table.append([n_mc, n_total - n_mc])
        mc_rows.append({
            "model": model_name,
            "cluster": c,
            "n_total": n_total,
            "n_with_metacritic": n_mc,
            "coverage_pct": fmt(100.0 * n_mc / n_total if n_total else 0.0, 2),
            "mean": fmt(float(mc_vals.mean()), 2) if n_mc else "",
            "std": fmt(float(mc_vals.std(ddof=0)), 2) if n_mc else "",
            "median": fmt(float(np.median(mc_vals)), 2) if n_mc else "",
            "q25": fmt(float(np.percentile(mc_vals, 25)), 2) if n_mc else "",
            "q75": fmt(float(np.percentile(mc_vals, 75)), 2) if n_mc else "",
            "min": fmt(float(mc_vals.min()), 2) if n_mc else "",
            "max": fmt(float(mc_vals.max()), 2) if n_mc else "",
        })

    mc_tests = {
        "coverage_chi2": chi2_from_table(np.array(mc_cov_table, dtype=int)),
        "score_comparison": compare_groups_numeric(mc_groups),
    }

    # --- 2. SteamSpy (muestra completa y robustez con Metacritic) ---
    ss_summary_rows = []
    ss_dist_rows = []
    ss_tests = {}

    for scope in ("all", "with_metacritic"):
        rank_groups = []
        dist_table = []
        for c in clusters:
            c_ids = [
                aid for aid, cl in assignments.items()
                if cl == c and (scope == "all" or sample_data[aid]["metacritic"] is not None)
            ]
            parsed_items = [
                steamspy_data[aid]["owners_parsed"]
                for aid in c_ids
                if aid in steamspy_data and steamspy_data[aid]["owners_parsed"] is not None
            ]
            ranks = np.array([tier_to_rank[item[2]] for item in parsed_items], dtype=float)
            lows = np.array([item[0] for item in parsed_items], dtype=float)
            rank_groups.append(ranks)

            n_total = len(c_ids)
            n_ss = len(parsed_items)
            counts = Counter(item[2] for item in parsed_items)
            dist_table.append([counts[label] for _, _, label in tier_order])

            for _, _, label in tier_order:
                cnt = counts[label]
                ss_dist_rows.append({
                    "model": model_name,
                    "scope": scope,
                    "cluster": c,
                    "owners_range": label,
                    "tier_rank": tier_to_rank[label],
                    "count": cnt,
                    "pct_within_cluster": fmt(100.0 * cnt / n_ss if n_ss else 0.0, 2),
                })

            if n_ss > 0:
                q25_r = int(np.percentile(ranks, 25, method="lower"))
                med_r = int(np.percentile(ranks, 50, method="lower"))
                q75_r = int(np.percentile(ranks, 75, method="lower"))
                q25_lbl, med_lbl, q75_lbl = rank_to_label[q25_r], rank_to_label[med_r], rank_to_label[q75_r]
                pct_100k = 100.0 * float((lows >= 100_000).mean())
                pct_500k = 100.0 * float((lows >= 500_000).mean())
                pct_1m = 100.0 * float((lows >= 1_000_000).mean())
            else:
                q25_lbl = med_lbl = q75_lbl = ""
                pct_100k = pct_500k = pct_1m = None

            ss_summary_rows.append({
                "model": model_name,
                "scope": scope,
                "cluster": c,
                "n_total": n_total,
                "n_with_steamspy": n_ss,
                "coverage_pct": fmt(100.0 * n_ss / n_total if n_total else 0.0, 2),
                "q25_tier": q25_lbl,
                "median_tier": med_lbl,
                "q75_tier": q75_lbl,
                "pct_100k_plus": fmt(pct_100k, 2),
                "pct_500k_plus": fmt(pct_500k, 2),
                "pct_1m_plus": fmt(pct_1m, 2),
            })

        ss_tests[scope] = {
            "ordinal_tier_comparison": compare_groups_numeric(rank_groups),
            "distribution_chi2": chi2_from_table(np.array(dist_table, dtype=int)),
        }

    tests_summary = {"metacritic": mc_tests, "steamspy": ss_tests}
    return mc_rows, ss_summary_rows, ss_dist_rows, tests_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--model-a", type=Path, default=DEFAULT_MODEL_A)
    parser.add_argument("--model-a-sub", type=Path, default=DEFAULT_MODEL_A_SUB)
    parser.add_argument("--model-b", type=Path, default=DEFAULT_MODEL_B)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        sample_data = read_sample_metacritic(args.sample)
        steamspy_data = read_steamspy(args.raw_dir, set(sample_data))
        models = {
            "modelo_A": read_assignments(args.model_a),
            "modelo_A_sub": read_assignments(args.model_a_sub),
            "modelo_B": read_assignments(args.model_b),
        }
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    tier_order = sorted(
        {
            info["owners_parsed"]
            for info in steamspy_data.values()
            if info["owners_parsed"] is not None
        },
        key=lambda item: (item[0], item[1]),
    )
    if not tier_order:
        parser.exit(1, "Error: no se encontraron rangos de propietarios válidos en steamspy*.jsonl.\n")
    tier_to_rank = {label: idx for idx, (_, _, label) in enumerate(tier_order)}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset cruzado para figuras y auditoría
    merged_rows = []
    for appid in sorted(sample_data, key=int):
        mc = sample_data[appid]["metacritic"]
        ss = steamspy_data.get(appid, {})
        parsed = ss.get("owners_parsed")
        merged_rows.append({
            "appid": appid,
            "name": sample_data[appid]["name"],
            "cluster_A": models["modelo_A"].get(appid, ""),
            "cluster_A_sub": models["modelo_A_sub"].get(appid, ""),
            "cluster_B": models["modelo_B"].get(appid, ""),
            "metacritic": "" if mc is None else f"{mc:.0f}",
            "has_metacritic": int(mc is not None),
            "steamspy_owners": parsed[2] if parsed else "",
            "owners_low": parsed[0] if parsed else "",
            "owners_high": parsed[1] if parsed else "",
            "owners_mid": int((parsed[0] + parsed[1]) // 2) if parsed else "",
            "owners_tier_rank": tier_to_rank[parsed[2]] if parsed else "",
        })

    write_csv(
        args.output_dir / "external_dataset.csv",
        list(merged_rows[0].keys()),
        merged_rows,
    )

    all_mc_rows, all_ss_summary, all_ss_dist = [], [], []
    all_tests = {}
    for model_name, assignments in models.items():
        mc_rows, ss_sum, ss_dist, tests = evaluate_model(
            model_name, assignments, sample_data, steamspy_data, tier_order
        )
        all_mc_rows.extend(mc_rows)
        all_ss_summary.extend(ss_sum)
        all_ss_dist.extend(ss_dist)
        all_tests[model_name] = tests

    write_csv(
        args.output_dir / "metacritic_summary.csv",
        list(all_mc_rows[0].keys()),
        all_mc_rows,
    )
    write_csv(
        args.output_dir / "steamspy_summary.csv",
        list(all_ss_summary[0].keys()),
        all_ss_summary,
    )
    write_csv(
        args.output_dir / "steamspy_distribution.csv",
        list(all_ss_dist[0].keys()),
        all_ss_dist,
    )

    n_sample = len(sample_data)
    n_mc_total = sum(1 for v in sample_data.values() if v["metacritic"] is not None)
    n_ss_total = sum(1 for v in steamspy_data.values() if v["owners_parsed"] is not None)

    summary = {
        "sample_rows": n_sample,
        "metacritic_coverage_global": {
            "count": n_mc_total,
            "pct": round(100.0 * n_mc_total / n_sample, 2),
        },
        "steamspy_coverage_global": {
            "count": n_ss_total,
            "pct": round(100.0 * n_ss_total / n_sample, 2),
        },
        "steamspy_tiers_ordered": [label for _, _, label in tier_order],
        "note": (
            "Metacritic y SteamSpy no intervienen en el clustering. Los rangos de SteamSpy "
            "se analizan como categorías ordinales estimadas, no como cifras exactas de ventas."
        ),
        "tests": all_tests,
    }
    with (args.output_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Reporte legible por consola
    print(f"Muestra total: {n_sample}")
    print(f"Cobertura Metacritic: {n_mc_total}/{n_sample} ({100.0 * n_mc_total / n_sample:.1f}%)")
    print(f"Cobertura SteamSpy:   {n_ss_total}/{n_sample} ({100.0 * n_ss_total / n_sample:.1f}%)\n")

    for model_name in models:
        print(f"=== {model_name} ===")
        print("Metacritic:")
        for r in [x for x in all_mc_rows if x["model"] == model_name]:
            print(
                f"  Cluster {r['cluster']}: n={r['n_total']} | con nota={r['n_with_metacritic']} "
                f"({r['coverage_pct']}%) | media={r['mean']} | mediana={r['median']} "
                f"[{r['q25']}-{r['q75']}]"
            )
        t_mc = all_tests[model_name]["metacritic"]
        p_cov = t_mc["coverage_chi2"]["p_value"]
        p_sc = t_mc["score_comparison"]["p_value"]
        r_sc = t_mc["score_comparison"].get("rank_biserial_r")
        print(
            f"  Tests Metacritic -> cobertura chi2 p={fmt(p_cov, 4)} | "
            f"puntuación MW-U p={fmt(p_sc, 4)}, r={fmt(r_sc, 3)}"
        )

        for scope, title in (("all", "SteamSpy (todos)"), ("with_metacritic", "SteamSpy (robustez: solo con Metacritic)")):
            print(f"{title}:")
            for r in [x for x in all_ss_summary if x["model"] == model_name and x["scope"] == scope]:
                print(
                    f"  Cluster {r['cluster']}: n={r['n_with_steamspy']}/{r['n_total']} "
                    f"({r['coverage_pct']}%) | mediana=[{r['median_tier']}] | "
                    f">=100k: {r['pct_100k_plus']}% | >=1M: {r['pct_1m_plus']}%"
                )
            t_ss = all_tests[model_name]["steamspy"][scope]
            p_ord = t_ss["ordinal_tier_comparison"]["p_value"]
            r_ord = t_ss["ordinal_tier_comparison"].get("rank_biserial_r")
            print(f"  Test SteamSpy ({scope}) -> ordinal MW-U p={fmt(p_ord, 4)}, r={fmt(r_ord, 3)}")
        print()

    print(f"Resultados guardados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()