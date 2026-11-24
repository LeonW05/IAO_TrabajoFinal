"""Prepara las variables del modelo B (metadatos + texto) sin ejecutar clustering."""

import argparse
import csv
import html
import importlib.util
import json
import math
import re
import statistics
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = PROJECT_ROOT / "data/interim/final_sample.csv"
DEFAULT_RAW = PROJECT_ROOT / "data/raw"
DEFAULT_REVIEWS = DEFAULT_RAW / "reviews.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/modelo_B"
MODELO_A_SCRIPT = PROJECT_ROOT / "scripts/modelo_A/prepare_features.py"

SEED = 20261109
MIN_REVIEWS = 10
SVD_COMPONENTS = 50
REFERENCE_DATE = date(2026, 11, 21)
# Campos de descripción por orden de preferencia (el primero no vacío tras limpiar).
DESCRIPTION_FIELDS = ("detailed_description", "about_the_game", "short_description")
# Peso por defecto de cada bloque (1 = misma varianza total que los metadatos).
BLOCK_WEIGHTS = {"metadata": 1.0, "reviews_text": 1.0, "description_text": 1.0, "review_stats": 0.5}

TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"
URL = re.compile(r"https?://\S+|www\.\S+")
BBCODE = re.compile(r"\[/?[A-Za-z0-9*]+(?:=[^\]]*)?\]")
HTML_TAG = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")
WORD = re.compile(r"[^\W\d_]+")


def load_modelo_a():
    """Reutiliza el preprocesado de metadatos del modelo A (mismo código, misma validación)."""
    spec = importlib.util.spec_from_file_location("modelo_a_prepare_features", MODELO_A_SCRIPT)
    if spec is None or spec.loader is None:
        raise ValueError(f"No se puede cargar {MODELO_A_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_review(text):
    text = URL.sub(" ", html.unescape(text))
    return SPACES.sub(" ", BBCODE.sub(" ", text)).strip()


def clean_description(text):
    text = html.unescape(HTML_TAG.sub(" ", text))
    return SPACES.sub(" ", URL.sub(" ", text)).strip()


def read_reviews(path, appids):
    """Devuelve, por appid, las reseñas con texto y el resumen de Steam (último registro válido)."""
    records = {}
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON inválido en {path}:{line_number}") from error
            appid = str(record.get("appid", ""))
            if appid not in appids:
                continue
            response = (record.get("payload") or {}).get("response")
            if record.get("status") != 200 or not isinstance(response, dict):
                continue
            summary = response.get("query_summary")
            if not isinstance(summary, dict):
                continue
            texts = [clean_review(item["review"]) for item in response.get("reviews") or []
                     if isinstance(item.get("review"), str)]
            records[appid] = {"texts": [text for text in texts if text], "summary": summary}
    missing = sorted(appid for appid in appids if appid not in records)
    if missing:
        raise ValueError(f"Faltan reseñas válidas para {len(missing)} juegos: {', '.join(missing[:10])}")
    return records


def read_descriptions(metadata, appids):
    texts, sources = {}, {}
    for appid in appids:
        for field in DESCRIPTION_FIELDS:
            value = metadata[appid].get(field)
            text = clean_description(value) if isinstance(value, str) else ""
            if text:
                texts[appid], sources[appid] = text, field
                break
        else:
            raise ValueError(f"appid {appid}: no hay descripción en {', '.join(DESCRIPTION_FIELDS)}.")
    return texts, sources


def review_statistics(appid, record):
    summary = record["summary"]
    positive, negative = summary.get("total_positive"), summary.get("total_negative")
    if type(positive) is not int or type(negative) is not int or positive + negative <= 0:
        raise ValueError(f"appid {appid}: query_summary sin totales de reseñas positivas/negativas.")
    words = statistics.mean(len(WORD.findall(text)) for text in record["texts"])
    return positive / (positive + negative), math.log1p(words)


def reduce_svd(matrix, components, seed):
    if not 1 <= components < min(matrix.shape):
        raise ValueError(f"--svd-components debe estar entre 1 y {min(matrix.shape) - 1}.")
    svd = TruncatedSVD(n_components=components, algorithm="randomized", random_state=seed)
    return normalize(svd.fit_transform(matrix)), svd


def standardize(values):
    mean, std = statistics.mean(values), statistics.pstdev(values)
    scale = std or 1.0
    return [(value - mean) / scale for value in values], {"mean": mean, "std": std, "scale": scale}


def build_features(sample, metadata, reviews, args, modelo_a):
    meta_rows, meta_config = modelo_a.prepare(sample, metadata, args.reference_date)
    appids = [row["appid"] for row in sample]
    descriptions, description_sources = read_descriptions(metadata, appids)

    # Texto de reseñas: un documento por juego, concatenando sus reseñas únicas.
    docs, reviews_used, duplicates_removed = [], 0, 0
    for appid in appids:
        texts = reviews[appid]["texts"]
        unique = list(dict.fromkeys(texts))
        duplicates_removed += len(texts) - len(unique)
        reviews_used += len(unique)
        docs.append(" ".join(unique))
    review_vectorizer = TfidfVectorizer(stop_words="english", token_pattern=TOKEN_PATTERN, sublinear_tf=True,
                                        min_df=5, max_df=0.5, max_features=20000)
    review_tfidf = normalize(review_vectorizer.fit_transform(docs))
    review_svd, review_svd_model = reduce_svd(review_tfidf, args.svd_components, args.seed)

    description_vectorizer = TfidfVectorizer(stop_words="english", token_pattern=TOKEN_PATTERN,
                                             sublinear_tf=True, min_df=3, max_df=0.5, max_features=20000)
    description_tfidf = normalize(description_vectorizer.fit_transform([descriptions[a] for a in appids]))
    description_svd, description_svd_model = reduce_svd(description_tfidf, args.svd_components, args.seed)

    stats = [review_statistics(appid, reviews[appid]) for appid in appids]
    positive, positive_params = standardize([item[0] for item in stats])
    length, length_params = standardize([item[1] for item in stats])

    width = len(str(args.svd_components))
    meta_columns = meta_config["feature_columns"]
    blocks = {
        "metadata": ([f"meta_{column}" for column in meta_columns],
                     np.array([[row[column] for column in meta_columns] for row in meta_rows], dtype=float)),
        "reviews_text": ([f"rev_svd_{i + 1:0{width}d}" for i in range(args.svd_components)], review_svd),
        "description_text": ([f"desc_svd_{i + 1:0{width}d}" for i in range(args.svd_components)],
                             description_svd),
        "review_stats": (["stats_pct_positive_scaled", "stats_review_words_log1p_scaled"],
                         np.column_stack([positive, length])),
    }
    # Cada bloque se lleva a la varianza total de los metadatos y se multiplica por su peso.
    target = blocks["metadata"][1].var(axis=0).sum()
    block_info, columns, parts = {}, [], []
    for name, (names, matrix) in blocks.items():
        variance = matrix.var(axis=0).sum()
        if variance <= 0:
            raise ValueError(f"El bloque {name} no tiene varianza.")
        weight = args.weights[name]
        scale = weight * math.sqrt(target / variance)
        block_info[name] = {"columns": len(names), "total_variance_before": variance, "weight": weight,
                            "block_scale": scale, "total_variance_after": variance * scale ** 2}
        columns += names
        parts.append(matrix * scale)
    final = np.hstack(parts)
    rows = [{"appid": appid, **dict(zip(columns, map(float, values)))} for appid, values in zip(appids, final)]

    config = {
        "reference_date": args.reference_date.isoformat(),
        "rows": len(rows),
        "feature_columns": columns,
        "identifier_column": "appid",
        "row_order": "appid ascendente; igual en features.csv y en las matrices tfidf_*.npz",
        "min_reviews": args.min_reviews,
        "seed": args.seed,
        "blocks": block_info,
        "block_scaling": "bloque * peso * sqrt(varianza_total_metadatos / varianza_total_bloque)",
        "metadata": {key: value for key, value in meta_config.items()
                     if key not in ("feature_columns", "identifier_column", "rows")},
        "metadata_note": "Mismo preprocesado que el modelo A, ajustado solo a los juegos incluidos.",
        "review_text": {
            "documents": len(docs), "reviews_used": reviews_used,
            "duplicates_removed": duplicates_removed,
            "vocabulary": len(review_vectorizer.vocabulary_),
            "aggregation": "un documento por juego concatenando sus reseñas únicas, normalizado L2",
            "svd_components": args.svd_components,
            "svd_explained_variance": float(review_svd_model.explained_variance_ratio_.sum()),
            "cleaning": "entidades HTML, URL y etiquetas BBCode/Steam eliminadas",
            "tfidf": "stop words inglesas, tokens alfabéticos >=2, sublinear_tf, min_df=5, max_df=0.5 "
                     "sobre documentos por juego, max_features=20000",
        },
        "description_text": {
            "documents": len(appids), "vocabulary": len(description_vectorizer.vocabulary_),
            "svd_components": args.svd_components,
            "svd_explained_variance": float(description_svd_model.explained_variance_ratio_.sum()),
            "field_priority": list(DESCRIPTION_FIELDS),
            "field_used": {field: sum(source == field for source in description_sources.values())
                           for field in DESCRIPTION_FIELDS},
            "tfidf": "stop words inglesas, tokens alfabéticos >=2, sublinear_tf, min_df=3, max_df=0.5, "
                     "max_features=20000",
        },
        "review_stats": {
            "pct_positive": "total_positive / (total_positive + total_negative) de query_summary (todas las "
                            "reseñas del juego, no solo las descargadas), estandarizado",
            "review_words": "log1p(media de palabras alfabéticas por reseña descargada), estandarizado",
            "scaling": {"pct_positive": positive_params, "review_words_log1p": length_params},
        },
        "svd": "TruncatedSVD aleatorizado, filas normalizadas L2 después de proyectar",
        "sklearn_version": sklearn.__version__,
        "excluded_columns": ["name", "metacritic", "SteamSpy", "stratum", "total_reviews"],
    }
    artifacts = {
        "tfidf_reviews": review_tfidf, "tfidf_description": description_tfidf,
        "vocabulary": {"reviews": review_vectorizer.get_feature_names_out().tolist(),
                       "description": description_vectorizer.get_feature_names_out().tolist()},
        "models": {"reviews_vectorizer": review_vectorizer, "reviews_svd": review_svd_model,
                   "description_vectorizer": description_vectorizer, "description_svd": description_svd_model},
    }
    return rows, config, artifacts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW, help="Carpeta con appdetails_candidates*.jsonl.")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-date", type=date.fromisoformat, default=REFERENCE_DATE,
                        help="Fecha de referencia AAAA-MM-DD (por defecto: 2026-11-21).")
    parser.add_argument("--min-reviews", type=int, default=MIN_REVIEWS,
                        help="Mínimo de reseñas con texto por juego (por defecto: 10).")
    parser.add_argument("--svd-components", type=int, default=SVD_COMPONENTS,
                        help="Componentes SVD por bloque de texto (por defecto: 50).")
    parser.add_argument("--seed", type=int, default=SEED)
    for name, default in BLOCK_WEIGHTS.items():
        parser.add_argument(f"--weight-{name.replace('_', '-')}", type=float, default=default,
                            dest=f"weight_{name}", help=f"Peso del bloque {name} (por defecto: {default}).")
    args = parser.parse_args()
    args.weights = {name: getattr(args, f"weight_{name}") for name in BLOCK_WEIGHTS}
    if args.min_reviews < 1 or any(weight < 0 for weight in args.weights.values()):
        parser.error("--min-reviews debe ser >= 1 y los pesos no pueden ser negativos.")
    return args


def main():
    args = parse_args()
    try:
        modelo_a = load_modelo_a()
        sample = modelo_a.read_sample(args.sample)
        appids = {row["appid"] for row in sample}
        reviews = read_reviews(args.reviews, appids)
        excluded = {appid: len(reviews[appid]["texts"]) for appid in sorted(appids, key=int)
                    if len(reviews[appid]["texts"]) < args.min_reviews}
        sample = [row for row in sample if row["appid"] not in excluded]
        if not sample:
            raise ValueError("Ningún juego alcanza el mínimo de reseñas.")
        metadata = modelo_a.read_metadata(args.raw_dir, {row["appid"] for row in sample})
        rows, config, artifacts = build_features(sample, metadata, reviews, args, modelo_a)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}")

    config.update({"sample_path": str(args.sample.resolve()), "raw_dir": str(args.raw_dir.resolve()),
                   "reviews_path": str(args.reviews.resolve()), "sample_rows": len(excluded) + len(rows),
                   "excluded_below_min_reviews": excluded})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "features.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["appid"] + config["feature_columns"])
        writer.writeheader()
        writer.writerows(rows)
    with (out / "preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with (out / "text_vocabulary.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts["vocabulary"], f, ensure_ascii=False)
    sp.save_npz(out / "tfidf_reviews.npz", artifacts["tfidf_reviews"].tocsr())
    sp.save_npz(out / "tfidf_description.npz", artifacts["tfidf_description"].tocsr())
    joblib.dump(artifacts["models"], out / "text_models.joblib")

    print(f"Juegos: {len(rows)} de {config['sample_rows']} | excluidos (<{args.min_reviews} reseñas): {len(excluded)}")
    print(f"Variables: {len(config['feature_columns'])} | fecha de referencia: {args.reference_date}")
    for name, info in config["blocks"].items():
        print(f"  {name}: {info['columns']} columnas | varianza final {info['total_variance_after']:.2f}")
    print(f"Vocabulario reseñas: {config['review_text']['vocabulary']} | descripción: "
          f"{config['description_text']['vocabulary']}")
    print(f"Varianza explicada SVD: reseñas {config['review_text']['svd_explained_variance']:.1%} | "
          f"descripción {config['description_text']['svd_explained_variance']:.1%}")
    print(f"Resultados: {out.resolve()}")


if __name__ == "__main__":
    main()