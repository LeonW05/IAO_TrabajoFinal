"""Compara las particiones de los modelos A y B sobre los mismos juegos.

Filtra las asignaciones de A a los appids presentes en B, comprueba que ambos
modelos cubran exactamente esa muestra, y guarda A_sub, B, una tabla cruzada,
el ARI y un resumen JSON.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A = PROJECT_ROOT / "modelos/modelo_A/cluster_assignments.csv"
DEFAULT_B = PROJECT_ROOT / "modelos/modelo_B/cluster_assignments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "modelos/comparacion_A_B"


def read_assignments(path, model_name):
    """Lee asignaciones y valida IDs y etiquetas sin asumir el orden de filas."""
    try:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["appid", "cluster"]:
                raise ValueError(f"{path}: se esperan las columnas appid,cluster.")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"No se puede leer el archivo de {model_name}: {path}") from error

    if not rows:
        raise ValueError(f"El archivo de asignaciones de {model_name} está vacío: {path}")

    assignments = {}
    for line_number, row in enumerate(rows, 2):
        appid, cluster = row["appid"], row["cluster"]
        if not appid or not appid.isdecimal():
            raise ValueError(f"{path}:{line_number}: appid inválido: {appid!r}.")
        if cluster is None or not cluster.isdecimal():
            raise ValueError(f"{path}:{line_number}: cluster inválido: {cluster!r}.")
        if appid in assignments:
            raise ValueError(f"{path}:{line_number}: appid duplicado: {appid}.")
        assignments[appid] = int(cluster)
    return assignments


def ordered_ids(assignments):
    return sorted(assignments, key=int)


def validate_labels(assignments, model_name):
    labels = set(assignments.values())
    if len(labels) < 2:
        raise ValueError(f"El modelo {model_name} debe contener al menos dos clusters.")
    if labels != set(range(len(labels))):
        raise ValueError(
            f"Las etiquetas del modelo {model_name} deben ser consecutivas desde 0; "
            f"se encontraron {sorted(labels)}."
        )


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare(assignments_a, assignments_b):
    """Devuelve asignaciones alineadas, tabla cruzada y ARI de A_sub frente a B."""
    ids_a, ids_b = set(assignments_a), set(assignments_b)
    missing_a = sorted(ids_b - ids_a, key=int)
    extra_a = sorted(ids_a - ids_b, key=int)
    if missing_a:
        raise ValueError(
            f"A no contiene {len(missing_a)} appids presentes en B; "
            f"ejemplos: {', '.join(missing_a[:10])}."
        )
    if not assignments_b:
        raise ValueError("El modelo B no contiene asignaciones.")

    common_ids = ordered_ids(assignments_b)
    a_sub = {appid: assignments_a[appid] for appid in common_ids}
    b_sub = {appid: assignments_b[appid] for appid in common_ids}
    validate_labels(a_sub, "A_sub")
    validate_labels(b_sub, "B")

    labels_a = sorted(set(a_sub.values()))
    labels_b = sorted(set(b_sub.values()))
    table = {label_a: Counter() for label_a in labels_a}
    for appid in common_ids:
        table[a_sub[appid]][b_sub[appid]] += 1

    ari = float(adjusted_rand_score(
        [a_sub[appid] for appid in common_ids],
        [b_sub[appid] for appid in common_ids],
    ))
    return common_ids, a_sub, b_sub, labels_a, labels_b, table, ari, extra_a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, default=DEFAULT_A,
                        help=f"Asignaciones de A (por defecto: {DEFAULT_A}).")
    parser.add_argument("--model-b", type=Path, default=DEFAULT_B,
                        help=f"Asignaciones de B (por defecto: {DEFAULT_B}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Directorio de resultados (por defecto: {DEFAULT_OUTPUT}).")
    args = parser.parse_args()

    try:
        assignments_a = read_assignments(args.model_a, "A")
        assignments_b = read_assignments(args.model_b, "B")
        (common_ids, a_sub, b_sub, labels_a, labels_b,
         table, ari, extra_a) = compare(assignments_a, assignments_b)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        a_sub_path = args.output_dir / "A_sub_assignments.csv"
        comparison_path = args.output_dir / "comparison_assignments.csv"
        crosstab_path = args.output_dir / "crosstab_A_vs_B.csv"
        summary_path = args.output_dir / "comparison.json"

        write_csv(a_sub_path, ["appid", "cluster"], [
            {"appid": appid, "cluster": a_sub[appid]} for appid in common_ids
        ])
        write_csv(comparison_path, ["appid", "cluster_A_sub", "cluster_B"], [
            {"appid": appid, "cluster_A_sub": a_sub[appid], "cluster_B": b_sub[appid]}
            for appid in common_ids
        ])
        write_csv(crosstab_path, ["cluster_A_sub", *[f"B_{label}" for label in labels_b], "total"], [
            {
                "cluster_A_sub": label_a,
                **{f"B_{label_b}": table[label_a][label_b] for label_b in labels_b},
                "total": sum(table[label_a].values()),
            }
            for label_a in labels_a
        ] + [{
            "cluster_A_sub": "total",
            **{
                f"B_{label_b}": sum(table[label_a][label_b] for label_a in labels_a)
                for label_b in labels_b
            },
            "total": len(common_ids),
        }])

        summary = {
            "model_a_path": str(args.model_a.resolve()),
            "model_b_path": str(args.model_b.resolve()),
            "rows_model_a": len(assignments_a),
            "rows_model_b": len(assignments_b),
            "rows_compared": len(common_ids),
            "a_only_rows_excluded": len(extra_a),
            "cluster_labels_a_sub": labels_a,
            "cluster_labels_b": labels_b,
            "adjusted_rand_index": ari,
            "metric": "Adjusted Rand Index (ARI)",
            "outputs": {
                "a_sub_assignments": str(a_sub_path.resolve()),
                "comparison_assignments": str(comparison_path.resolve()),
                "crosstab": str(crosstab_path.resolve()),
            },
        }
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Juegos comparados: {len(common_ids)}")
    print(f"Juegos de A excluidos por no estar en B: {len(extra_a)}")
    print(f"ARI (A_sub vs B): {ari:.6f}")
    print(f"Resultados en {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
