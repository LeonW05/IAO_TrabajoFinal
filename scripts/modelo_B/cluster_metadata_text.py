"""Compara K-Means y Ward para distintos números de grupos en el modelo B.

Lee features.csv y preprocessing.json (metadatos + texto), prueba ambos
algoritmos para k=2..10, calcula silhouette y Davies-Bouldin sobre la
partición completa y exporta las métricas, las asignaciones y los perfiles.
Los perfiles incluyen los términos TF-IDF más característicos de cada grupo.
Persiste en `.pkl` únicamente K-Means cuando resulta seleccionado; Ward no
ofrece predicción para juegos nuevos.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data/processed/modelo_B"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modelos/modelo_B"
MODELO_A_SCRIPT = PROJECT_ROOT / "scripts/modelo_A/cluster_metadata.py"
SEED = 20261109
TERM_LIMIT = 10
PROFILE_FIELDS = [
    "cluster", "size", "share", "mean_price_eur", "mean_age_years",
    "rate_platform_windows", "rate_platform_mac", "rate_platform_linux",
    "mean_pct_positive", "mean_review_words",
    "top_genres", "top_categories", "top_review_terms", "top_description_terms",
]


def load_modelo_a():
    """Reutiliza las funciones genéricas del clustering del modelo A."""
    spec = importlib.util.spec_from_file_location("modelo_a_cluster_metadata", MODELO_A_SCRIPT)
    if spec is None or spec.loader is None:
        raise ValueError(f"No se puede cargar {MODELO_A_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_text(output_dir, reviews_path, description_path, vocabulary_path, rows):
    """Carga las matrices TF-IDF y el vocabulario, alineados con las filas de features.csv."""
    paths = {
        "reviews": reviews_path or output_dir / "tfidf_reviews.npz",
        "description": description_path or output_dir / "tfidf_description.npz",
    }
    matrices = {}
    for name, path in paths.items():
        if not path.exists():
            raise ValueError(f"Falta {path}; ejecuta prepare_features.py antes del clustering.")
        matrix = sp.load_npz(path).tocsr()
        if matrix.shape[0] != rows:
            raise ValueError(f"{path} tiene {matrix.shape[0]} filas y features.csv {rows}.")
        matrices[name] = matrix
    vocabulary_path = vocabulary_path or output_dir / "text_vocabulary.json"
    try:
        with vocabulary_path.open(encoding="utf-8") as f:
            vocabulary = json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON inválido en {vocabulary_path}: {error}") from error
    if not isinstance(vocabulary, dict):
        raise ValueError(f"{vocabulary_path} no contiene un diccionario de vocabularios.")
    for name, matrix in matrices.items():
        terms = vocabulary.get(name)
        if not isinstance(terms, list) or len(terms) != matrix.shape[1]:
            raise ValueError(f"El vocabulario {name} no coincide con {matrix.shape[1]} columnas.")
    return {"matrices": matrices, "vocabulary": vocabulary}


def top_terms(matrix, mask, terms, limit):
    """Términos con mayor peso medio en el grupo, ignorando pesos no positivos."""
    if len(mask) != matrix.shape[0]:
        raise ValueError("La máscara no coincide con las filas de la matriz.")
    if len(terms) != matrix.shape[1]:
        raise ValueError("El vocabulario no coincide con las columnas de la matriz.")
    if not mask.any():
        return ""
    centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
    ranked = sorted(
        ((float(weight), terms[index]) for index, weight in enumerate(centroid) if weight > 0),
        key=lambda item: (-item[0], item[1]),
    )
    return "; ".join(f"{term} ({weight:.3f})" for weight, term in ranked[:limit])


def build_profiles(matrix, labels, feature_columns, config, n_clusters, text, modelo_a):
    metadata = config.get("metadata") or {}
    vocabularies = metadata.get("vocabularies") or {}
    scaling = metadata.get("numeric_scaling") or {}
    try:
        price_mean = float(scaling["price_log1p"]["mean"])
        price_scale = float(scaling["price_log1p"]["scale"])
        age_mean = float(scaling["age_years"]["mean"])
        age_scale = float(scaling["age_years"]["scale"])
        pct_params = config["review_stats"]["scaling"]["pct_positive"]
        pct_mean, pct_scale = float(pct_params["mean"]), float(pct_params["scale"])
        words_params = config["review_stats"]["scaling"]["review_words_log1p"]
        words_mean, words_scale = float(words_params["mean"]), float(words_params["scale"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "preprocessing.json no contiene los parámetros de escalado del modelo B."
        ) from None
    names = {
        "price": "meta_price_log1p_scaled",
        "age": "meta_age_years_scaled",
        "windows": "meta_platform_windows",
        "mac": "meta_platform_mac",
        "linux": "meta_platform_linux",
        "pct": "stats_pct_positive_scaled",
        "words": "stats_review_words_log1p_scaled",
    }
    missing = [name for name in names.values() if name not in feature_columns]
    if missing:
        raise ValueError(f"Faltan columnas para describir los grupos: {', '.join(missing)}")
    labels = np.asarray(labels)
    if len(labels) != matrix.shape[0]:
        raise ValueError("Las etiquetas no coinciden con las filas de la matriz.")
    genres = modelo_a.column_indexes(feature_columns, "meta_genres_")
    categories = modelo_a.column_indexes(feature_columns, "meta_categories_")
    profiles = []
    total = len(labels)
    for cluster in range(n_clusters):
        mask = labels == cluster
        size = int(mask.sum())
        if size == 0:
            profiles.append({
                "cluster": cluster, "size": 0, "share": 0.0,
                "mean_price_eur": 0.0, "mean_age_years": 0.0,
                "rate_platform_windows": 0.0, "rate_platform_mac": 0.0,
                "rate_platform_linux": 0.0, "mean_pct_positive": 0.0,
                "mean_review_words": 0.0, "top_genres": "", "top_categories": "",
                "top_review_terms": "", "top_description_terms": "",
            })
            continue
        points = matrix[mask]
        price_log1p = points[:, feature_columns.index(names["price"])] * price_scale + price_mean
        age_years = points[:, feature_columns.index(names["age"])] * age_scale + age_mean
        pct_values = points[:, feature_columns.index(names["pct"])] * pct_scale + pct_mean
        words_log = points[:, feature_columns.index(names["words"])] * words_scale + words_mean
        profiles.append({
            "cluster": cluster,
            "size": size,
            "share": size / total,
            "mean_price_eur": float(np.expm1(price_log1p).mean()),
            "mean_age_years": float(age_years.mean()),
            "rate_platform_windows": modelo_a.mean_column(points, feature_columns, names["windows"]),
            "rate_platform_mac": modelo_a.mean_column(points, feature_columns, names["mac"]),
            "rate_platform_linux": modelo_a.mean_column(points, feature_columns, names["linux"]),
            "mean_pct_positive": float(pct_values.mean() * 100),
            "mean_review_words": float(np.expm1(words_log).mean()),
            "top_genres": modelo_a.top_rates(points, genres, vocabularies.get("genres") or {}),
            "top_categories": modelo_a.top_rates(points, categories, vocabularies.get("categories") or {}),
            "top_review_terms": top_terms(text["matrices"]["reviews"], mask,
                                          text["vocabulary"]["reviews"], TERM_LIMIT),
            "top_description_terms": top_terms(text["matrices"]["description"], mask,
                                               text["vocabulary"]["description"], TERM_LIMIT),
        })
    return profiles


def save_outputs(output_dir, features_path, preprocessing_path, appids, results, selected,
                 labels, profiles, k_values, n_features, seeds, best_model, config, modelo_a):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "clustering_metrics.csv"
    assignments_path = output_dir / "cluster_assignments.csv"
    profiles_path = output_dir / "cluster_profiles.csv"
    summary_path = output_dir / "clustering.json"
    model_path = output_dir / "modelo_B.pkl"
    metric_rows = [{
        "algorithm": row["algorithm"],
        "k": row["k"],
        "n_init": row["n_init"],
        "silhouette": modelo_a.number(row["silhouette"]),
        "davies_bouldin": modelo_a.number(row["davies_bouldin"]),
        "stability_ari": modelo_a.number(row["stability_ari"]),
        "selected": int(row is selected),
    } for row in results]
    modelo_a.write_csv(
        metrics_path,
        ["algorithm", "k", "n_init", "silhouette", "davies_bouldin", "stability_ari", "selected"],
        metric_rows,
    )
    modelo_a.write_csv(
        assignments_path,
        ["appid", "cluster"],
        [{"appid": appid, "cluster": cluster} for appid, cluster in zip(appids, labels)],
    )
    profile_rows = [{
        **profile,
        "share": modelo_a.number(profile["share"]),
        "mean_price_eur": modelo_a.number(profile["mean_price_eur"]),
        "mean_age_years": modelo_a.number(profile["mean_age_years"]),
        "rate_platform_windows": modelo_a.number(profile["rate_platform_windows"]),
        "rate_platform_mac": modelo_a.number(profile["rate_platform_mac"]),
        "rate_platform_linux": modelo_a.number(profile["rate_platform_linux"]),
        "mean_pct_positive": modelo_a.number(profile["mean_pct_positive"]),
        "mean_review_words": modelo_a.number(profile["mean_review_words"]),
    } for profile in profiles]
    modelo_a.write_csv(profiles_path, PROFILE_FIELDS, profile_rows)
    summary = {
        "features_path": str(features_path.resolve()),
        "preprocessing_path": str(preprocessing_path.resolve()),
        "rows": len(appids),
        "n_features": n_features,
        "k_values": list(k_values),
        "seeds": list(seeds),
        "algorithms": list(modelo_a.ALGORITHMS),
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
        "feature_blocks": config.get("blocks"),
        "text_terms_limit": TERM_LIMIT,
        "profile_units": {
            "mean_price_eur": "EUR",
            "mean_age_years": "años",
            "rate_platform_*": "proporción 0-1",
            "mean_pct_positive": "porcentaje 0-100",
            "mean_review_words": "palabras por reseña",
        },
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


def print_report(results, selected, output_dir, modelo_a):
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
            f"{modelo_a.number(row['stability_ari']):>11}"
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
    parser.add_argument("--tfidf-reviews", type=Path, default=None,
                        help="Por defecto, tfidf_reviews.npz dentro de --output-dir.")
    parser.add_argument("--tfidf-description", type=Path, default=None,
                        help="Por defecto, tfidf_description.npz dentro de --output-dir.")
    parser.add_argument("--vocabulary", type=Path, default=None,
                        help="Por defecto, text_vocabulary.json dentro de --output-dir.")
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
        modelo_a = load_modelo_a()
        appids, raw_matrix, feature_columns, config = modelo_a.read_inputs(
            args.features, args.preprocessing
        )
        if args.k_max >= len(appids):
            raise ValueError(
                f"k-max={args.k_max} no cabe en una muestra de {len(appids)} juegos."
            )
        matrix = modelo_a.validate_matrix(raw_matrix)
        text = load_text(args.output_dir, args.tfidf_reviews, args.tfidf_description,
                         args.vocabulary, len(appids))
        k_values = range(args.k_min, args.k_max + 1)
        results, models, labels_by_configuration = [], {}, {}
        for algorithm in modelo_a.ALGORITHMS:
            for k in k_values:
                if algorithm == "kmeans":
                    fitted = [modelo_a.fit_model(algorithm, matrix, k, seed) for seed in args.seeds]
                    model = fitted[0]
                    seed_labels = [fit.labels_ for fit in fitted]
                    stability_ari = float(np.mean([
                        adjusted_rand_score(seed_labels[first], seed_labels[second])
                        for first in range(len(seed_labels))
                        for second in range(first + 1, len(seed_labels))
                    ]))
                else:
                    model = modelo_a.fit_model(algorithm, matrix, k)
                    stability_ari = None
                labels_k, _ = modelo_a.relabel_by_size(appids, model.labels_, k)
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
                labels_by_configuration[(algorithm, k)] = labels_k
                if algorithm == "kmeans":
                    models[(algorithm, k)] = model
        if not results:
            raise ValueError(
                "Ninguna combinación produjo grupos con al menos dos juegos. "
                "Reduce k-max o revisa la muestra."
            )
        selected = modelo_a.select_best(results)
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
        profiles = build_profiles(matrix, labels, feature_columns, config,
                                  selected["k"], text, modelo_a)
        save_outputs(
            args.output_dir, args.features, args.preprocessing, appids, results, selected,
            labels, profiles, k_values, len(feature_columns), args.seeds, best_model,
            config, modelo_a,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Juegos: {len(appids)} | variables: {len(feature_columns)}")
    print_report(results, selected, args.output_dir, modelo_a)


if __name__ == "__main__":
    main()
