"""Genera las figuras de análisis interno, comparación A x B y validación externa
para la memoria del proyecto (modelos/figuras/*.png).
"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "modelos" / "figuras"

# Estilo limpio y homogéneo para memoria académica
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PALETTE = ["#2b5c8f", "#d95f02"]  # Cluster 0 (azul), Cluster 1 (naranja)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_top_list(raw: str) -> list[tuple[str, float]]:
    """Convierte 'Indie (67%); Action (38%)' o 'boss (0.019); ...' en lista de tuplas."""
    if not raw:
        return []
    items = []
    for part in raw.split(";"):
        part = part.strip()
        if not part or "(" not in part or not part.endswith(")"):
            continue
        label, val_str = part.rsplit("(", 1)
        val_str = val_str[:-1].strip()
        try:
            if val_str.endswith("%"):
                val = float(val_str[:-1]) / 100.0
            else:
                val = float(val_str)
            items.append((label.strip(), val))
        except ValueError:
            continue
    return items


def load_game_level_data():
    """Reconstruye precio en EUR, antigüedad en años y asignaciones de A y B por appid."""
    with (PROJECT_ROOT / "data/processed/modelo_A/preprocessing.json").open(encoding="utf-8") as f:
        prep_a = json.load(f)
    p_mean = prep_a["numeric_scaling"]["price_log1p"]["mean"]
    p_scale = prep_a["numeric_scaling"]["price_log1p"]["scale"]
    a_mean = prep_a["numeric_scaling"]["age_years"]["mean"]
    a_scale = prep_a["numeric_scaling"]["age_years"]["scale"]

    meta_by_id = {}
    for row in read_csv_rows(PROJECT_ROOT / "data/processed/modelo_A/features.csv"):
        aid = row["appid"]
        p_log = float(row["price_log1p_scaled"]) * p_scale + p_mean
        age = float(row["age_years_scaled"]) * a_scale + a_mean
        meta_by_id[aid] = {"price_eur": math.expm1(p_log), "age_years": age}

    assign_a = {r["appid"]: int(r["cluster"]) for r in read_csv_rows(PROJECT_ROOT / "modelos/modelo_A/cluster_assignments.csv")}
    assign_b = {r["appid"]: int(r["cluster"]) for r in read_csv_rows(PROJECT_ROOT / "modelos/modelo_B/cluster_assignments.csv")}
    return meta_by_id, assign_a, assign_b


# ---------------------------------------------------------------------------
# FIGURA 1: Selección de k y algoritmo (Silhouette y Davies-Bouldin)
# ---------------------------------------------------------------------------
def plot_model_selection():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    configs = [
        ("Modelo A (Metadatos)", PROJECT_ROOT / "modelos/modelo_A/clustering_metrics.csv", 0),
        ("Modelo B (Metadatos + Texto)", PROJECT_ROOT / "modelos/modelo_B/clustering_metrics.csv", 1),
    ]
    algo_style = {
        "kmeans": ("K-Means", "o-", "#2b5c8f"),
        "agglomerative": ("Ward (aglomerativo)", "s--", "#d95f02"),
    }

    for title, path, col in configs:
        rows = read_csv_rows(path)
        for algo, (label, style, color) in algo_style.items():
            sub = sorted([r for r in rows if r["algorithm"] == algo], key=lambda x: int(x["k"]))
            ks = [int(r["k"]) for r in sub]
            sil = [float(r["silhouette"]) for r in sub]
            db = [float(r["davies_bouldin"]) for r in sub]
            axes[0, col].plot(ks, sil, style, color=color, label=label, linewidth=1.8, markersize=5)
            axes[1, col].plot(ks, db, style, color=color, label=label, linewidth=1.8, markersize=5)

            for r in sub:
                if r.get("selected") == "1":
                    k_sel = int(r["k"])
                    axes[0, col].scatter([k_sel], [float(r["silhouette"])], s=130, facecolors="none", edgecolors="crimson", linewidths=2, zorder=5)
                    axes[1, col].scatter([k_sel], [float(r["davies_bouldin"])], s=130, facecolors="none", edgecolors="crimson", linewidths=2, zorder=5)

        axes[0, col].set_title(title)
        axes[1, col].set_xlabel("Número de grupos (k)")
        axes[1, col].set_xticks(range(2, 11))

    axes[0, 0].set_ylabel("Coeficiente Silhouette (mayor = mejor)")
    axes[1, 0].set_ylabel("Índice Davies-Bouldin (menor = mejor)")
    axes[0, 1].legend(loc="upper right", frameon=True)

    fig.suptitle("Evolución de métricas internas según k y algoritmo (círculo rojo = seleccionado)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_seleccion_k_metricas.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURA 2: Heatmap de perfiles de clusters (Modelo A y Modelo B)
# ---------------------------------------------------------------------------
def plot_profiles_heatmap():
    prof_a = read_csv_rows(PROJECT_ROOT / "modelos/modelo_A/cluster_profiles.csv")
    prof_b = read_csv_rows(PROJECT_ROOT / "modelos/modelo_B/cluster_profiles.csv")

    # Recopilar proporciones exactas de géneros desde features.csv de A y B
    def get_genre_rates(features_path, prep_path, assignments, prefix):
        with prep_path.open(encoding="utf-8") as f:
            prep = json.load(f)
        vocab = prep["metadata"]["vocabularies"]["genres"] if "metadata" in prep else prep["vocabularies"]["genres"]
        rows = read_csv_rows(features_path)
        by_c = {0: [], 1: []}
        for r in rows:
            c = assignments[r["appid"]]
            by_c[c].append(r)
        rates = {}
        for gid, gname in vocab.items():
            col = f"{prefix}{gid}"
            if col in rows[0]:
                rates[gname] = [np.mean([float(x[col]) for x in by_c[c]]) for c in (0, 1)]
        return rates

    assign_a = {r["appid"]: int(r["cluster"]) for r in read_csv_rows(PROJECT_ROOT / "modelos/modelo_A/cluster_assignments.csv")}
    assign_b = {r["appid"]: int(r["cluster"]) for r in read_csv_rows(PROJECT_ROOT / "modelos/modelo_B/cluster_assignments.csv")}

    g_a = get_genre_rates(PROJECT_ROOT / "data/processed/modelo_A/features.csv", PROJECT_ROOT / "data/processed/modelo_A/preprocessing.json", assign_a, "genres_")
    g_b = get_genre_rates(PROJECT_ROOT / "data/processed/modelo_B/features.csv", PROJECT_ROOT / "data/processed/modelo_B/preprocessing.json", assign_b, "meta_genres_")

    key_genres = ["Indie", "Action", "Adventure", "Casual", "Simulation", "RPG", "Strategy"]
    row_labels = [
        "Precio medio (€)",
        "Antigüedad media (años)",
        "Soporte macOS (%)",
        "Soporte Linux (%)",
        "% Reseñas positivas",
        "Palabras / reseña",
        *[f"Género: {g} (%)" for g in key_genres],
    ]

    # Matriz de valores reales para anotar en celdas
    raw_matrix = np.full((len(row_labels), 4), np.nan)
    for c in (0, 1):
        raw_matrix[0, c] = float(prof_a[c]["mean_price_eur"])
        raw_matrix[1, c] = float(prof_a[c]["mean_age_years"])
        raw_matrix[2, c] = float(prof_a[c]["rate_platform_mac"]) * 100
        raw_matrix[3, c] = float(prof_a[c]["rate_platform_linux"]) * 100
        for idx, g in enumerate(key_genres):
            raw_matrix[6 + idx, c] = g_a.get(g, [0, 0])[c] * 100

        raw_matrix[0, 2 + c] = float(prof_b[c]["mean_price_eur"])
        raw_matrix[1, 2 + c] = float(prof_b[c]["mean_age_years"])
        raw_matrix[2, 2 + c] = float(prof_b[c]["rate_platform_mac"]) * 100
        raw_matrix[3, 2 + c] = float(prof_b[c]["rate_platform_linux"]) * 100
        raw_matrix[4, 2 + c] = float(prof_b[c]["mean_pct_positive"])
        raw_matrix[5, 2 + c] = float(prof_b[c]["mean_review_words"])
        for idx, g in enumerate(key_genres):
            raw_matrix[6 + idx, 2 + c] = g_b.get(g, [0, 0])[c] * 100

    # Normalización por fila [0, 1] para colorear el contraste relativo
    norm_matrix = np.zeros_like(raw_matrix)
    for i in range(raw_matrix.shape[0]):
        row = raw_matrix[i]
        valid = row[~np.isnan(row)]
        vmin, vmax = valid.min(), valid.max()
        if vmax > vmin:
            norm_matrix[i] = (row - vmin) / (vmax - vmin)
        else:
            norm_matrix[i] = 0.5

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#eeeeee")

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    im = ax.imshow(np.ma.masked_invalid(norm_matrix), aspect="auto", cmap=cmap, vmin=0, vmax=1)

    col_labels = [
        f"Modelo A · C0\n(n={prof_a[0]['size']})",
        f"Modelo A · C1\n(n={prof_a[1]['size']})",
        f"Modelo B · C0\n(n={prof_b[0]['size']})",
        f"Modelo B · C1\n(n={prof_b[1]['size']})",
    ]
    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.grid(False)
    ax.axvline(1.5, color="black", linewidth=2)

    for i in range(raw_matrix.shape[0]):
        for j in range(4):
            val = raw_matrix[i, j]
            if np.isnan(val):
                txt = "N/A"
                color = "#666666"
            else:
                txt = f"{val:.1f} €" if i == 0 else (f"{val:.1f}" if i in (1, 5) else f"{val:.1f}%")
                color = "white" if norm_matrix[i, j] > 0.62 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9, fontweight="medium")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_label("Intensidad relativa por fila (mín -> máx)")
    ax.set_title("Perfil comparado de los clusters: Modelo A (metadatos) vs. Modelo B (+ texto)", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_heatmap_perfiles.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURA 3: Distribución de precio y antigüedad por cluster (A y B)
# ---------------------------------------------------------------------------
def plot_price_age_distributions():
    meta_by_id, assign_a, assign_b = load_game_level_data()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    models = [("Modelo A (Ward, k=2)", assign_a, 0), ("Modelo B (K-Means, k=2)", assign_b, 1)]

    for title, assigns, col in models:
        p0 = [meta_by_id[a]["price_eur"] for a, c in assigns.items() if c == 0]
        p1 = [meta_by_id[a]["price_eur"] for a, c in assigns.items() if c == 1]
        a0 = [meta_by_id[a]["age_years"] for a, c in assigns.items() if c == 0]
        a1 = [meta_by_id[a]["age_years"] for a, c in assigns.items() if c == 1]

        bp_p = axes[0, col].boxplot(
            [p0, p1],
            tick_labels=[f"Cluster 0\n(n={len(p0)})", f"Cluster 1\n(n={len(p1)})"],
            patch_artist=True,
            showfliers=False,
            widths=0.45,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, color in zip(bp_p["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[0, col].set_title(title)

        bp_a = axes[1, col].boxplot(
            [a0, a1],
            tick_labels=[f"Cluster 0\n(n={len(a0)})", f"Cluster 1\n(n={len(a1)})"],
            patch_artist=True,
            showfliers=False,
            widths=0.45,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, color in zip(bp_a["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    axes[0, 0].set_ylabel("Precio original (EUR, sin outliers extremos)")
    axes[1, 0].set_ylabel("Antigüedad desde lanzamiento (años)")
    fig.suptitle("Distribución de precio y antigüedad por cluster", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_precio_antiguedad_clusters.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURA 4: Términos TF-IDF más representativos del Modelo B
# ---------------------------------------------------------------------------
def plot_tfidf_terms():
    prof_b = read_csv_rows(PROJECT_ROOT / "modelos/modelo_B/cluster_profiles.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    configs = [
        (0, "top_review_terms", "Cluster 0 (Acción/Combate) — Reseñas", PALETTE[0], 0, 0),
        (1, "top_review_terms", "Cluster 1 (Puzles/Casual/Novela) — Reseñas", PALETTE[1], 0, 1),
        (0, "top_description_terms", "Cluster 0 (Acción/Combate) — Descripción", PALETTE[0], 1, 0),
        (1, "top_description_terms", "Cluster 1 (Puzles/Casual/Novela) — Descripción", PALETTE[1], 1, 1),
    ]

    for c_idx, field, title, color, r, c in configs:
        terms = parse_top_list(prof_b[c_idx][field])[::-1]
        words = [t[0] for t in terms]
        weights = [t[1] for t in terms]
        ax = axes[r, c]
        ax.barh(words, weights, color=color, alpha=0.85, height=0.68)
        ax.set_title(title)
        ax.set_xlabel("Peso TF-IDF medio en el grupo")
        ax.set_xlim(0, max(weights) * 1.18 if weights else 0.03)
        for idx, w in enumerate(weights):
            ax.text(w + max(weights) * 0.02, idx, f"{w:.3f}", va="center", fontsize=8.5)

    fig.suptitle("Top 10 términos TF-IDF por grupo en el Modelo B", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_tfidf_terminos_modelo_B.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURA 5: Comparación A_sub vs B (matriz de contingencia y ARI)
# ---------------------------------------------------------------------------
def plot_comparison_a_vs_b():
    rows = read_csv_rows(PROJECT_ROOT / "modelos/comparacion_A_B/crosstab_A_vs_B.csv")
    with (PROJECT_ROOT / "modelos/comparacion_A_B/comparison.json").open(encoding="utf-8") as f:
        comp = json.load(f)

    data_rows = [r for r in rows if r["cluster_A_sub"] != "total"]
    mat = np.array([[int(r["B_0"]), int(r["B_1"])] for r in data_rows], dtype=float)
    row_pct = mat / mat.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Modelo B · Cluster 0\n(Acción/Combate)", "Modelo B · Cluster 1\n(Puzles/Casual)"], fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Modelo A_sub · Cluster 0\n(Antiguos/Baratos)", "Modelo A_sub · Cluster 1\n(Recientes/Caros)"], fontweight="bold")
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            cnt = int(mat[i, j])
            pct = row_pct[i, j]
            color = "white" if pct > 55 else "black"
            ax.text(j, i, f"{cnt} juegos\n({pct:.1f}% fila)", ha="center", va="center", color=color, fontsize=10.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% dentro de cada grupo de A_sub")
    ari = comp["adjusted_rand_index"]
    ax.set_title(f"Cruce de asignaciones A_sub × B (n={comp['rows_compared']}, ARI = {ari:.4f})", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_comparacion_A_vs_B.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURA 6: Validación externa (SteamSpy + Metacritic)
# ---------------------------------------------------------------------------
def plot_external_validation():
    ext_rows = read_csv_rows(PROJECT_ROOT / "modelos/validacion_externa/external_dataset.csv")
    ss_dist = read_csv_rows(PROJECT_ROOT / "modelos/validacion_externa/steamspy_distribution.csv")
    mc_sum = read_csv_rows(PROJECT_ROOT / "modelos/validacion_externa/metacritic_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))

    # Fila superior: distribución de rangos SteamSpy (todos) para A y B
    for col, (model_key, title) in enumerate([("modelo_A", "Modelo A — Propietarios SteamSpy (todos)"), ("modelo_B", "Modelo B — Propietarios SteamSpy (todos)")]):
        sub = [r for r in ss_dist if r["model"] == model_key and r["scope"] == "all"]
        tiers = sorted({(int(r["tier_rank"]), r["owners_range"]) for r in sub})
        x = np.arange(len(tiers))
        width = 0.38

        for c_idx in (0, 1):
            c_map = {r["owners_range"]: float(r["pct_within_cluster"]) for r in sub if int(r["cluster"]) == c_idx}
            vals = [c_map.get(lbl, 0.0) for _, lbl in tiers]
            offset = -width / 2 if c_idx == 0 else width / 2
            axes[0, col].bar(x + offset, vals, width=width, color=PALETTE[c_idx], alpha=0.85, label=f"Cluster {c_idx}")

        short_labels = [lbl.replace(",000", "k").replace(" .. ", "-") for _, lbl in tiers]
        axes[0, col].set_xticks(x)
        axes[0, col].set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("% juegos dentro del cluster")
        axes[0, col].legend(frameon=True)

    # Fila inferior izquierda: Cobertura Metacritic (%) en A y B
    labels_cov = ["Modelo A\nCluster 0", "Modelo A\nCluster 1", "Modelo B\nCluster 0", "Modelo B\nCluster 1"]
    cov_vals = []
    for m_key, c_idx in [("modelo_A", 0), ("modelo_A", 1), ("modelo_B", 0), ("modelo_B", 1)]:
        row = next(r for r in mc_sum if r["model"] == m_key and int(r["cluster"]) == c_idx)
        cov_vals.append(float(row["coverage_pct"]))

    bars = axes[1, 0].bar(labels_cov, cov_vals, color=[PALETTE[0], PALETTE[1], PALETTE[0], PALETTE[1]], alpha=0.85, width=0.52)
    axes[1, 0].axvline(1.5, color="gray", linestyle=":", linewidth=1.2)
    axes[1, 0].set_ylabel("% juegos con nota en Metacritic")
    axes[1, 0].set_title("Cobertura de crítica (Metacritic) por cluster")
    axes[1, 0].set_ylim(0, max(cov_vals) * 1.22)
    for bar, v in zip(bars, cov_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Fila inferior derecha: Boxplot puntuaciones Metacritic
    mc_groups = [
        [float(r["metacritic"]) for r in ext_rows if r["cluster_A"] == "0" and r["metacritic"]],
        [float(r["metacritic"]) for r in ext_rows if r["cluster_A"] == "1" and r["metacritic"]],
        [float(r["metacritic"]) for r in ext_rows if r["cluster_B"] == "0" and r["metacritic"]],
        [float(r["metacritic"]) for r in ext_rows if r["cluster_B"] == "1" and r["metacritic"]],
    ]
    bp = axes[1, 1].boxplot(
        mc_groups,
        tick_labels=[
            f"Mod. A · C0\n(n={len(mc_groups[0])})",
            f"Mod. A · C1\n(n={len(mc_groups[1])})",
            f"Mod. B · C0\n(n={len(mc_groups[2])})",
            f"Mod. B · C1\n(n={len(mc_groups[3])})",
        ],
        patch_artist=True,
        widths=0.48,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], [PALETTE[0], PALETTE[1], PALETTE[0], PALETTE[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1, 1].axvline(2.5, color="gray", linestyle=":", linewidth=1.2)
    axes[1, 1].set_ylabel("Puntuación Metacritic (0–100)")
    axes[1, 1].set_title("Distribución de notas de Metacritic (subconjunto con nota)")

    fig.suptitle("Validación externa: popularidad comercial (SteamSpy) y crítica (Metacritic)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_validacion_externa.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_selection()
    plot_profiles_heatmap()
    plot_price_age_distributions()
    plot_tfidf_terms()
    plot_comparison_a_vs_b()
    plot_external_validation()
    print(f"Figuras generadas en {OUT_DIR.resolve()}:")
    for path in sorted(OUT_DIR.glob("*.png")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()