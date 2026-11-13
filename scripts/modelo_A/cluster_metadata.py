"""Prueba algoritmos y números de grupos sobre las variables del modelo A.

Lee features.csv y preprocessing.json, compara k-means y clustering
aglomerativo, y guarda métricas, asignaciones y perfiles del modelo elegido.
No modifica la muestra ni las variables de entrada.
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data/processed/modelo_A"
SEED = 20261109
DEFAULT_SEEDS = tuple(SEED + offset for offset in range(5))
ALGORITHMS = ("kmeans", "agglomerative")
PROFILE_LIMIT = 5


def load_estimators():
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
    except ImportError as error:
        raise SystemExit(
            "Faltan numpy o scikit-learn. Instálalos en el entorno con: "
            "pip install numpy scikit-learn"
        ) from error
    return {
        "np": np,
        "KMeans": KMeans,
        "AgglomerativeClustering": AgglomerativeClustering,
        "adjusted_rand_score": adjusted_rand_score,
        "davies_bouldin_score": davies_bouldin_score,
        "silhouette_score": silhouette_score,
    }


def read_inputs(features_path, preprocessing_path):
    try:
        with preprocessing_path.open(encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON inválido en {preprocessing_path}: {error}") from error
    feature_columns = config.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("preprocessing.json no contiene feature_columns.")
    if config.get("identifier_column", "appid") != "appid":
        raise ValueError("La columna identificadora debe ser appid y no entra en el clustering.")
    with features_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = ["appid", *feature_columns]
        if reader.fieldnames != expected:
            raise ValueError("Las columnas de features.csv no coinciden con preprocessing.json.")
        rows = list(reader)
    appids = [row["appid"] for row in rows]
    if not rows or any(not value.isdecimal() for value in appids):
        raise ValueError("features.csv está vacío o contiene appids inválidos.")
    if len(set(appids)) != len(appids):
        raise ValueError("features.csv contiene appids duplicados.")
    if config.get("rows") not in (None, len(rows)):
        raise ValueError("El número de filas no coincide con preprocessing.json.")
    order = sorted(range(len(rows)), key=lambda index: int(appids[index]))
    if order != list(range(len(rows))):
        rows = [rows[index] for index in order]
        appids = [row["appid"] for row in rows]
    matrix = []
    for row in rows:
        try:
            matrix.append([float(row[column]) for column in feature_columns])
        except ValueError as error:
            raise ValueError(f"Valor no numérico en appid {row['appid']}.") from error
    return appids, matrix, feature_columns, config


def validate_matrix(np, matrix):
    values = np.asarray(matrix, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Hay valores no finitos en las variables.")
    return values


def fit_labels(tools, algorithm, matrix, k, seed):
    if algorithm == "kmeans":
        model = tools["KMeans"](n_clusters=k, n_init=10, random_state=seed, algorithm="lloyd")
        return model.fit_predict(matrix)
    if algorithm == "agglomerative":
        model = tools["AgglomerativeClustering"](n_clusters=k, linkage="ward")
        return model.fit_predict(matrix)
    raise ValueError(f"Algoritmo desconocido: {algorithm}")


def partition_metrics(tools, matrix, labels):
    labels = tools["np"].asarray(labels)
    counts = tools["np"].bincount(labels)
    if counts.size < 2 or counts.min() < 2:
        return None, None
    silhouette = float(tools["silhouette_score"](matrix, labels, metric="euclidean"))
    davies_bouldin = float(tools["davies_bouldin_score"](matrix, labels))
    return silhouette, davies_bouldin


def mean_pairwise_ari(tools, labelings):
    score = tools["adjusted_rand_score"]
    values = [
        float(score(left, right))
        for index, left in enumerate(labelings)
        for right in labelings[index + 1:]
    ]
    return sum(values) / len(values) if values else None


def inertia(np, matrix, labels):
    labels = np.asarray(labels)
    total = 0.0
    for label in np.unique(labels):
        points = matrix[labels == label]
        center = points.mean(axis=0)
        total += float(((points - center) ** 2).sum())
    return total


def relabel_by_size(appids, labels):
    groups = {}
    for appid, label in zip(appids, labels):
        groups.setdefault(int(label), []).append(appid)
    order = sorted(
        groups,
        key=lambda label: (-len(groups[label]), min(int(appid) for appid in groups[label])),
    )
    mapping = {old: new for new, old in enumerate(order)}
    return [mapping[int(label)] for label in labels]


def better(candidate, current):
    if candidate["silhouette"] != current["silhouette"]:
        return candidate["silhouette"] > current["silhouette"]
    if candidate["davies_bouldin"] != current["davies_bouldin"]:
        return candidate["davies_bouldin"] < current["davies_bouldin"]
    candidate_stability = -1 if candidate["stability_ari"] is None else candidate["stability_ari"]
    current_stability = -1 if current["stability_ari"] is None else current["stability_ari"]
    if candidate_stability != current_stability:
        return candidate_stability > current_stability
    if candidate["k"] != current["k"]:
        return candidate["k"] < current["k"]
    return candidate["algorithm"] < current["algorithm"]


def column_indexes(feature_columns, prefix):
    return [
        (index, column[len(prefix):])
        for index, column in enumerate(feature_columns)
        if column.startswith(prefix)
    ]


def top_rates(np, points, indexed_columns, vocabulary):
    ranked = []
    for index, identifier in indexed_columns:
        rate = float(points[:, index].mean())
        if rate <= 0:
            continue
        label = vocabulary.get(identifier, identifier) or identifier
        ranked.append((rate, label))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return "; ".join(f"{label} ({rate:.0%})" for rate, label in ranked[:PROFILE_LIMIT])


def mean_column(np, points, feature_columns, name):
    return float(points[:, feature_columns.index(name)].mean())


def build_profiles(np, matrix, labels, feature_columns, config):
    required = (
        "price_log1p_scaled", "age_years_scaled",
        "platform_windows", "platform_mac", "platform_linux",
    )
    missing = [name for name in required if name not in feature_columns]
    if missing:
        raise ValueError(f"Faltan columnas para describir los grupos: {', '.join(missing)}")
    vocabularies = config.get("vocabularies") or {}
    genres = column_indexes(feature_columns, "genres_")
    categories = column_indexes(feature_columns, "categories_")
    profiles = []
    total = len(labels)
    for cluster in range(max(labels) + 1):
        points = matrix[np.asarray(labels) == cluster]
        size = int(points.shape[0])
        profiles.append({
            "cluster": cluster,
            "size": size,
            "share": size / total,
            "mean_price_log1p_scaled": mean_column(np, points, feature_columns, "price_log1p_scaled"),
            "mean_age_years_scaled": mean_column(np, points, feature_columns, "age_years_scaled"),
            "rate_platform_windows": mean_column(np, points, feature_columns, "platform_windows"),
            "rate_platform_mac": mean_column(np, points, feature_columns, "platform_mac"),
            "rate_platform_linux": mean_column(np, points, feature_columns, "platform_linux"),
            "top_genres": top_rates(np, points, genres, vocabularies.get("genres") or {}),
            "top_categories": top_rates(np, points, categories, vocabularies.get("categories") or {}),
        })
    return profiles


def search(tools, appids, matrix, k_values, seeds):
    np = tools["np"]
    reference_seed = seeds[0]
    results = []
    chosen_labels = {}
    for algorithm in ALGORITHMS:
        for k in k_values:
            if algorithm == "agglomerative":
                labelings = [fit_labels(tools, algorithm, matrix, k, reference_seed)]
                stability = None
            else:
                labelings = [fit_labels(tools, algorithm, matrix, k, seed) for seed in seeds]
                stability = mean_pairwise_ari(tools, labelings)
            labels = relabel_by_size(appids, labelings[0])
            silhouette, davies_bouldin = partition_metrics(tools, matrix, labels)
            if silhouette is None:
                continue
            results.append({
                "algorithm": algorithm,
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "stability_ari": stability,
                "inertia": inertia(np, matrix, labels),
            })
            chosen_labels[(algorithm, k)] = labels
    if not results:
        raise ValueError(
            "Ninguna combinación produjo grupos con al menos dos juegos. "
            "Amplía la muestra o reduce el rango de k."
        )
    selected = results[0]
    for candidate in results[1:]:
        if better(candidate, selected):
            selected = candidate
    return results, selected, chosen_labels[(selected["algorithm"], selected["k"])]


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return "" if value is None else f"{value:.6f}"


def save_outputs(output_dir, features_path, preprocessing_path, appids, results, selected,
                 labels, profiles, k_values, seeds, n_features):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "clustering_metrics.csv"
    assignments_path = output_dir / "cluster_assignments.csv"
    profiles_path = output_dir / "cluster_profiles.csv"
    summary_path = output_dir / "clustering.json"
    metric_rows = []
    for row in results:
        metric_rows.append({
            "algorithm": row["algorithm"],
            "k": row["k"],
            "silhouette": number(row["silhouette"]),
            "davies_bouldin": number(row["davies_bouldin"]),
            "stability_ari": number(row["stability_ari"]),
            "inertia": number(row["inertia"]),
            "selected": int(row is selected),
        })
    write_csv(
        metrics_path,
        ["algorithm", "k", "silhouette", "davies_bouldin", "stability_ari", "inertia", "selected"],
        metric_rows,
    )
    write_csv(
        assignments_path,
        ["appid", "cluster"],
        [{"appid": appid, "cluster": cluster} for appid, cluster in zip(appids, labels)],
    )
    profile_fields = [
        "cluster", "size", "share", "mean_price_log1p_scaled", "mean_age_years_scaled",
        "rate_platform_windows", "rate_platform_mac", "rate_platform_linux",
        "top_genres", "top_categories",
    ]
    profile_rows = []
    for profile in profiles:
        profile_rows.append({
            **profile,
            "share": number(profile["share"]),
            "mean_price_log1p_scaled": number(profile["mean_price_log1p_scaled"]),
            "mean_age_years_scaled": number(profile["mean_age_years_scaled"]),
            "rate_platform_windows": number(profile["rate_platform_windows"]),
            "rate_platform_mac": number(profile["rate_platform_mac"]),
            "rate_platform_linux": number(profile["rate_platform_linux"]),
        })
    write_csv(profiles_path, profile_fields, profile_rows)
    summary = {
        "features_path": str(features_path.resolve()),
        "preprocessing_path": str(preprocessing_path.resolve()),
        "rows": len(appids),
        "n_features": n_features,
        "algorithms": list(ALGORITHMS),
        "k_values": list(k_values),
        "reference_seed": seeds[0],
        "stability_seeds": list(seeds),
        "kmeans": {"n_init": 10, "algorithm": "lloyd"},
        "agglomerative": {"linkage": "ward", "stability_ari": None},
        "metric": "euclidean",
        "selection_rule": (
            "Mayor silhouette; empate: menor Davies-Bouldin, luego mayor "
            "ARI medio entre semillas, luego menor k."
        ),
        "stability_note": (
            "El ARI medio solo se calcula en k-means. El aglomerativo Ward "
            "no usa semilla; su estabilidad queda vacía."
        ),
        "selected": {
            "algorithm": selected["algorithm"],
            "k": selected["k"],
            "silhouette": selected["silhouette"],
            "davies_bouldin": selected["davies_bouldin"],
            "stability_ari": selected["stability_ari"],
            "inertia": selected["inertia"],
        },
        "outputs": {
            "metrics": str(metrics_path.resolve()),
            "assignments": str(assignments_path.resolve()),
            "profiles": str(profiles_path.resolve()),
        },
        "excluded_from_fit": ["appid", "name", "metacritic", "reviews", "SteamSpy", "stratum"],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return metrics_path, assignments_path, profiles_path


def print_report(results, selected, output_dir):
    print(f"{'algoritmo':<16} {'k':>3} {'silhouette':>12} {'davies_bouldin':>16} {'estabilidad':>13}")
    for row in results:
        stability = "—" if row["stability_ari"] is None else f"{row['stability_ari']:.3f}"
        print(
            f"{row['algorithm']:<16} {row['k']:>3} {row['silhouette']:>12.3f} "
            f"{row['davies_bouldin']:>16.3f} {stability:>13}"
        )
    stability = "—" if selected["stability_ari"] is None else f"{selected['stability_ari']:.3f}"
    print(
        f"\nSeleccionado: {selected['algorithm']}, k={selected['k']}, "
        f"silhouette={selected['silhouette']:.3f}, "
        f"davies_bouldin={selected['davies_bouldin']:.3f}, estabilidad={stability}"
    )
    print(f"Resultados: {output_dir.resolve()}")


def parse_seeds(text):
    seeds = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece.isdecimal():
            raise ValueError("Las semillas deben ser enteros separados por comas.")
        seeds.append(int(piece))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Hacen falta al menos dos semillas distintas para medir la estabilidad.")
    return tuple(seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_DIR / "features.csv")
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_DIR / "preprocessing.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS),
                        help="Semillas de k-means separadas por comas. La primera es la de referencia.")
    args = parser.parse_args()
    try:
        if args.k_min < 2 or args.k_max < args.k_min:
            raise ValueError("El rango de grupos debe cumplir 2 <= k-min <= k-max.")
        seeds = parse_seeds(args.seeds)
        appids, raw_matrix, feature_columns, config = read_inputs(args.features, args.preprocessing)
        if args.k_max >= len(appids):
            raise ValueError(
                f"k-max={args.k_max} no cabe en una muestra de {len(appids)} juegos."
            )
        tools = load_estimators()
        matrix = validate_matrix(tools["np"], raw_matrix)
        k_values = range(args.k_min, args.k_max + 1)
        results, selected, labels = search(tools, appids, matrix, k_values, seeds)
        profiles = build_profiles(tools["np"], matrix, labels, feature_columns, config)
        save_outputs(
            args.output_dir, args.features, args.preprocessing, appids, results, selected,
            labels, profiles, k_values, seeds, len(feature_columns),
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print_report(results, selected, args.output_dir)


if __name__ == "__main__":
    main()
