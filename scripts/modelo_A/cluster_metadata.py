"""Compara K-Means y Ward para distintos números de grupos en el modelo A.

Lee features.csv y preprocessing.json, prueba ambos algoritmos para k=2..10,
calcula silhouette y Davies-Bouldin sobre la partición completa y exporta
las métricas, las asignaciones y los perfiles. Persiste en `.pkl` únicamente
K-Means cuando resulta seleccionado; Ward no ofrece predicción para juegos
nuevos.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data/processed/modelo_A"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modelos/modelo_A"
SEED = 20261109
ALGORITHMS = ("kmeans", "agglomerative")
ALGORITHM_RANK = {algorithm: rank for rank, algorithm in enumerate(ALGORITHMS)}
PROFILE_LIMIT = 5


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


def validate_matrix(matrix):
    values = np.asarray(matrix, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Hay valores no finitos en las variables.")
    return values


def fit_model(algorithm, matrix, k, seed=SEED):
    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=seed,
            algorithm="lloyd",
        )
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            metric="euclidean",
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")
    return model.fit(matrix)


def select_best(results):
    """Mayor silhouette; en empate, menor Davies-Bouldin, menor k y K-Means."""
    def key(row):
        db = row["davies_bouldin"]
        return (
            -row["silhouette"],
            float("inf") if db is None else db,
            row["k"],
            ALGORITHM_RANK[row["algorithm"]],
        )
    return min(results, key=key)


def relabel_by_size(appids, labels, n_clusters):
    groups = {}
    for appid, label in zip(appids, labels):
        groups.setdefault(int(label), []).append(appid)
    order = sorted(
        groups,
        key=lambda label: (-len(groups[label]), min(int(appid) for appid in groups[label])),
    )
    empty = [c for c in range(n_clusters) if c not in groups]
    order = order + empty
    mapping = {old: new for new, old in enumerate(order)}
    return [mapping[int(label)] for label in labels], order


def column_indexes(feature_columns, prefix):
    return [
        (index, column[len(prefix):])
        for index, column in enumerate(feature_columns)
        if column.startswith(prefix)
    ]


def top_rates(points, indexed_columns, vocabulary):
    ranked = []
    for index, identifier in indexed_columns:
        rate = float(points[:, index].mean())
        if rate <= 0:
            continue
        label = vocabulary.get(identifier, identifier) or identifier
        ranked.append((rate, label))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return "; ".join(f"{label} ({rate:.0%})" for rate, label in ranked[:PROFILE_LIMIT])


def mean_column(points, feature_columns, name):
    return float(points[:, feature_columns.index(name)].mean())


def build_profiles(matrix, labels, feature_columns, config, n_clusters):
    required = (
        "price_log1p_scaled", "age_years_scaled",
        "platform_windows", "platform_mac", "platform_linux",
    )
    missing = [name for name in required if name not in feature_columns]
    if missing:
        raise ValueError(f"Faltan columnas para describir los grupos: {', '.join(missing)}")
    vocabularies = config.get("vocabularies") or {}
    scaling = config.get("numeric_scaling") or {}
    try:
        price_mean = float(scaling["price_log1p"]["mean"])
        price_scale = float(scaling["price_log1p"]["scale"])
        age_mean = float(scaling["age_years"]["mean"])
        age_scale = float(scaling["age_years"]["scale"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "preprocessing.json no contiene el escalado de precio y antigüedad."
        ) from None
    genres = column_indexes(feature_columns, "genres_")
    categories = column_indexes(feature_columns, "categories_")
    profiles = []
    total = len(labels)
    for cluster in range(n_clusters):
        points = matrix[np.asarray(labels) == cluster]
        size = int(points.shape[0])
        if size == 0:
            profiles.append({
                "cluster": cluster,
                "size": 0,
                "share": 0.0,
                "mean_price_eur": 0.0,
                "mean_age_years": 0.0,
                "rate_platform_windows": 0.0,
                "rate_platform_mac": 0.0,
                "rate_platform_linux": 0.0,
                "top_genres": "",
                "top_categories": "",
            })
            continue
        price_log1p = (
            points[:, feature_columns.index("price_log1p_scaled")] * price_scale
            + price_mean
        )
        age_years = (
            points[:, feature_columns.index("age_years_scaled")] * age_scale
            + age_mean
        )
        profiles.append({
            "cluster": cluster,
            "size": size,
            "share": size / total,
            "mean_price_eur": float(np.expm1(price_log1p).mean()),
            "mean_age_years": float(age_years.mean()),
            "rate_platform_windows": mean_column(points, feature_columns, "platform_windows"),
            "rate_platform_mac": mean_column(points, feature_columns, "platform_mac"),
            "rate_platform_linux": mean_column(points, feature_columns, "platform_linux"),
            "top_genres": top_rates(points, genres, vocabularies.get("genres") or {}),
            "top_categories": top_rates(points, categories, vocabularies.get("categories") or {}),
        })
    return profiles


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return "" if value is None else f"{value:.6f}"


def save_outputs(output_dir, features_path, preprocessing_path, appids, results, selected,
                 labels, profiles, k_values, n_features, seeds, best_model):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "clustering_metrics.csv"
    assignments_path = output_dir / "cluster_assignments.csv"
    profiles_path = output_dir / "cluster_profiles.csv"
    summary_path = output_dir / "clustering.json"
    model_path = output_dir / "modelo_A.pkl"
    metric_rows = []
    for row in results:
        metric_rows.append({
            "algorithm": row["algorithm"],
            "k": row["k"],
            "n_init": row["n_init"],
            "silhouette": number(row["silhouette"]),
            "davies_bouldin": number(row["davies_bouldin"]),
            "stability_ari": number(row["stability_ari"]),
            "selected": int(row is selected),
        })
    write_csv(
        metrics_path,
        ["algorithm", "k", "n_init", "silhouette", "davies_bouldin", "stability_ari", "selected"],
        metric_rows,
    )
    write_csv(
        assignments_path,
        ["appid", "cluster"],
        [{"appid": appid, "cluster": cluster} for appid, cluster in zip(appids, labels)],
    )
    profile_fields = [
        "cluster", "size", "share", "mean_price_eur", "mean_age_years",
        "rate_platform_windows", "rate_platform_mac", "rate_platform_linux",
        "top_genres", "top_categories",
    ]
    profile_rows = []
    for profile in profiles:
        profile_rows.append({
            **profile,
            "share": number(profile["share"]),
            "mean_price_eur": number(profile["mean_price_eur"]),
            "mean_age_years": number(profile["mean_age_years"]),
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
        "k_values": list(k_values),
        "seeds": list(seeds),
        "algorithms": list(ALGORITHMS),
        "selection_rule": (
            "Mayor silhouette; empate: menor Davies-Bouldin; "
            "empate: menor k; empate: K-Means antes que Ward."
        ),
        "stability_note": (
            "ARI medio entre las 10 parejas de las 5 semillas de K-Means; "
            "no interviene en la selección."
        ),
        "selected": {
            "algorithm": selected["algorithm"],
            "n_init": selected["n_init"],
            "k": selected["k"],
            "silhouette": selected["silhouette"],
            "davies_bouldin": selected["davies_bouldin"],
            "predict_available": best_model is not None,
        },
        "metric_scope": "partición completa",
        "outputs": {
            "metrics": str(metrics_path.resolve()),
            "assignments": str(assignments_path.resolve()),
            "profiles": str(profiles_path.resolve()),
            "model": str(model_path.resolve()) if best_model is not None else None,
        },
        "excluded_from_fit": ["appid", "name", "metacritic", "reviews", "SteamSpy", "stratum"],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if best_model is not None:
        joblib.dump(best_model, model_path)
    elif model_path.exists():
        model_path.unlink()
    return metrics_path, assignments_path, profiles_path, model_path if best_model is not None else None


def print_report(results, selected, output_dir):
    print(
        f"{'algoritmo':>14} {'k':>3} {'n_init':>6} {'silhouette':>12} "
        f"{'davies_bouldin':>15} {'ARI medio':>11}"
    )
    for row in results:
        db = "—" if row["davies_bouldin"] is None else f"{row['davies_bouldin']:.3f}"
        n_init = "—" if row["n_init"] is None else str(row["n_init"])
        print(
            f"{row['algorithm']:>14} {row['k']:>3} {n_init:>6} "
            f"{row['silhouette']:>12.3f} {db:>15} "
            f"{number(row['stability_ari']):>11}"
        )
    db = "—" if selected["davies_bouldin"] is None else f"{selected['davies_bouldin']:.3f}"
    print(
        f"\nSeleccionado: algoritmo={selected['algorithm']}, k={selected['k']}, "
        f"n_init={selected['n_init']}, silhouette={selected['silhouette']:.3f}, "
        f"davies_bouldin={db}"
    )
    if selected["algorithm"] == "agglomerative":
        print(
            "Ward no ofrece predict() para juegos nuevos; "
            "no se guarda un modelo predictivo .pkl."
        )
    print(f"Resultados en {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_DIR / "features.csv")
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_DIR / "preprocessing.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument(
        "--seeds", type=int, nargs=5, default=[SEED + offset for offset in range(5)],
        metavar=("S1", "S2", "S3", "S4", "S5"),
        help="Cinco semillas para K-Means (por defecto: 20261109 a 20261113).",
    )

    args = parser.parse_args()
    try:
        if args.k_min < 2 or args.k_max < args.k_min:
            raise ValueError("El rango de grupos debe cumplir 2 <= k-min <= k-max.")
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Las cinco semillas deben ser distintas.")
        appids, raw_matrix, feature_columns, config = read_inputs(args.features, args.preprocessing)
        if args.k_max >= len(appids):
            raise ValueError(
                f"k-max={args.k_max} no cabe en una muestra de {len(appids)} juegos."
            )
        matrix = validate_matrix(raw_matrix)
        k_values = range(args.k_min, args.k_max + 1)
        results = []
        models = {}
        labels_by_configuration = {}
        for algorithm in ALGORITHMS:
            for k in k_values:
                if algorithm == "kmeans":
                    fitted_models = [fit_model(algorithm, matrix, k, seed) for seed in args.seeds]
                    model = fitted_models[0]
                    labels_by_seed = [fitted.labels_ for fitted in fitted_models]
                    stability_ari = float(np.mean([
                        adjusted_rand_score(labels_by_seed[first], labels_by_seed[second])
                        for first in range(len(labels_by_seed))
                        for second in range(first + 1, len(labels_by_seed))
                    ]))
                else:
                    model = fit_model(algorithm, matrix, k)
                    stability_ari = None

                labels_k, _ = relabel_by_size(appids, model.labels_, k)
                counts = np.bincount(labels_k)
                if counts.size < 2 or counts.min() < 2:
                    continue
                results.append({
                    "algorithm": algorithm,
                    "k": k,
                    "n_init": 10 if algorithm == "kmeans" else None,
                    "silhouette": float(silhouette_score(matrix, labels_k)),
                    "davies_bouldin": float(davies_bouldin_score(matrix, labels_k)),
                    "stability_ari": stability_ari,
                })
                key = (algorithm, k)
                labels_by_configuration[key] = labels_k
                if algorithm == "kmeans":
                    models[key] = model
        if not results:
            raise ValueError(
                "Ninguna combinación produjo grupos con al menos dos juegos. "
                "Reduce k-max o revisa la muestra."
            )
        selected = select_best(results)
        selected_key = (selected["algorithm"], selected["k"])
        labels = labels_by_configuration[selected_key]
        best_model = models.get(selected_key)
        if best_model is not None:
            cluster_order = sorted(
                range(best_model.n_clusters),
                key=lambda cluster: (
                    -int(np.sum(best_model.labels_ == cluster)),
                    min(int(appid) for appid, label in zip(appids, best_model.labels_)
                        if label == cluster),
                ),
            )
            best_model.cluster_centers_ = best_model.cluster_centers_[cluster_order]
            best_model.labels_ = np.asarray(labels, dtype=np.int32)

        profiles = build_profiles(matrix, labels, feature_columns, config, selected["k"])
        save_outputs(
            args.output_dir, args.features, args.preprocessing, appids, results, selected,
            labels, profiles, k_values, len(feature_columns), args.seeds, best_model,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print_report(results, selected, args.output_dir)


if __name__ == "__main__":
    main()
