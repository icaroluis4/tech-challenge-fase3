"""Fase 6 — Clusterização de municípios por perfil educacional e socioeconômico.

Escopo reduzido (D14): KMeans com k ∈ [3..8] escolhido por silhouette, sem
UMAP/mapa. O fit **exclui** metas, resultados de 2025 e UF/região — queremos
padrões estruturais, não geografia forçada nem leakage do target.

Validação externa *a posteriori*: taxa de atingimento da meta 2025 por cluster
(nunca entra no fit). Saídas:

- ``reports/clusters_municipios.csv`` — ``co_municipio, cluster, nome_cluster``
- ``reports/clusters_perfil.md`` — escolha de k, perfis e validação externa
- ``images/clusters_01_silhouette.png`` e ``images/clusters_02_atingimento.png``

Uso::

    python -m src.modeling.cluster_municipios
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import DATA_PROCESSED, IMAGES_DIR, REPORTS_DIR, SEED
from src.modeling.train_risco_meta import carregar_dados
from src.preprocessing.transformers import build_preprocessor, get_feature_names

log = logging.getLogger(__name__)

K_RANGE = range(3, 9)

# Metas/resultados (leakage do target) e geografia (não queremos clusters = UF).
# `atingiu_meta_2024` é derivada de resultado — também fora.
DROP_FIT = (
    "co_uf", "sg_uf", "regiao",
    "meta_2024", "meta_2025", "meta_2026", "meta_2027", "meta_2028",
    "meta_2029", "meta_2030", "gap_meta_2024", "esforco_2025",
    "ambicao_2030", "atingiu_meta_2024",
)

CSV_PATH = REPORTS_DIR / "clusters_municipios.csv"
MD_PATH = REPORTS_DIR / "clusters_perfil.md"


def montar_matriz_cluster(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Seleciona features estruturais para o fit (sem metas nem geografia).

    Devolve (X, num_cols, cat_cols); ids ficam fora de X (só no índice de saída).
    """
    ids = {"co_municipio", "no_municipio"}
    cols = [c for c in features.columns if c not in DROP_FIT and c not in ids]
    df = features[["co_municipio", "no_municipio", *cols]].copy()
    cat_cols = [c for c in cols if c in ("porte", "capital_uf")]
    num_cols = [c for c in cols if c not in cat_cols
                and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    x = df[num_cols + cat_cols]
    return x, num_cols, cat_cols


def escolher_k(x_transformed: np.ndarray, k_range=K_RANGE, seed: int = SEED) -> pd.DataFrame:
    """Roda KMeans para cada k e devolve tabela com silhouette e inércia."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(x_transformed)
        sil = silhouette_score(x_transformed, labels, sample_size=min(5_000, len(labels)),
                               random_state=seed)
        rows.append({"k": k, "silhouette": sil, "inercia": km.inertia_})
        log.info("k=%d → silhouette=%.4f | inércia=%.0f", k, sil, km.inertia_)
    return pd.DataFrame(rows)


def nomear_clusters(perfil: pd.DataFrame) -> dict[int, str]:
    """Nomeia clusters por regras sobre o perfil (resultado 2024, porte, rural).

    `perfil` tem uma linha por cluster com médias das features originais.
    """
    nomes: dict[int, str] = {}
    pc = perfil["pc_alfabetizado_2024"]
    rural = perfil["pct_escolas_rurais"]  # proporção 0–1
    pop = perfil["populacao_2024"]
    for c in perfil.index:
        nivel = "alto desempenho" if pc[c] >= pc.median() else "baixo desempenho"
        ruralidade = "rural" if rural[c] >= 0.5 else "urbano"
        if pop[c] >= 100_000:
            porte = "grande"
        elif pop[c] >= 20_000:
            porte = "médio"
        else:
            porte = "pequeno"
        nomes[c] = f"{nivel} {ruralidade} {porte}"
    return nomes


def _fig_silhouette(scores: pd.DataFrame, k_best: int):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(scores["k"], scores["silhouette"], "o-", color="C0", label="silhouette")
    ax1.axvline(k_best, ls="--", color="gray", alpha=0.7)
    ax1.set_xlabel("k")
    ax1.set_ylabel("silhouette", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["inercia"], "s--", color="C1", label="inércia")
    ax2.set_ylabel("inércia (elbow)", color="C1")
    ax1.set_title(f"Escolha de k — silhouette máximo em k={k_best}")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "clusters_01_silhouette.png", dpi=120)
    plt.close(fig)


def _fig_atingimento(taxa: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 4))
    (taxa * 100).plot.bar(ax=ax, color="C2")
    ax.axhline(72.5, ls="--", color="red", label="Brasil 72,5%")
    ax.set_ylabel("% atingiu meta 2025")
    ax.set_xlabel("cluster")
    ax.set_title("Validação externa — atingimento da meta 2025 por cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "clusters_02_atingimento.png", dpi=120)
    plt.close(fig)


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    features, targets = carregar_dados()

    x, num_cols, cat_cols = montar_matriz_cluster(features)
    log.info("matriz de cluster: %d×%d (%d num, %d cat)", x.shape[0], x.shape[1],
             len(num_cols), len(cat_cols))

    pre = build_preprocessor(num_cols, cat_cols, scale=True)
    xt = pre.fit_transform(x)
    log.info("após pré-processamento: %s (%d features)", xt.shape,
             len(get_feature_names(pre)))

    scores = escolher_k(xt)
    k_best = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    log.info("k escolhido: %d (silhouette=%.4f)", k_best,
             scores["silhouette"].max())

    km = KMeans(n_clusters=k_best, n_init=10, random_state=SEED)
    labels = km.fit_predict(xt)

    out = features[["co_municipio", "no_municipio"]].copy()
    out["cluster"] = labels

    # Perfil: médias das features originais por cluster.
    perfil = features.assign(cluster=labels).groupby("cluster").median(numeric_only=True)
    nomes = nomear_clusters(perfil)
    out["nome_cluster"] = out["cluster"].map(nomes)

    # Validação externa (nunca entrou no fit): atingimento 2025 e região.
    val = (out.merge(targets[["co_municipio", "atingiu_meta_2025"]], on="co_municipio")
              .dropna(subset=["atingiu_meta_2025"]))
    val["atingiu_meta_2025"] = val["atingiu_meta_2025"].astype(int)
    taxa = val.groupby("cluster")["atingiu_meta_2025"].mean()
    dist_regiao = (features.assign(cluster=labels)
                   .groupby(["cluster", "regiao"], observed=True).size()
                   .unstack(fill_value=0))

    out.to_csv(CSV_PATH, index=False)
    log.info("gravado %s (%d linhas)", CSV_PATH, len(out))

    _fig_silhouette(scores, k_best)
    _fig_atingimento(taxa)

    md = [
        "# Clusterização de municípios — Fase 6\n",
        "\nFit sem metas, sem resultados 2025 e sem UF/região (D14: escopo reduzido,",
        "sem UMAP/mapa). Pré-processamento idêntico ao dos modelos (mediana →",
        "symlog1p/scaler + OHE).\n",
        "\n## Escolha de k (silhouette)\n\n",
        scores.to_markdown(index=False, floatfmt=".4f"),
        f"\n\n**k escolhido: {k_best}** (maior silhouette).\n",
        "\n## Clusters nomeados\n\n",
        pd.DataFrame({"cluster": list(nomes), "nome_cluster": list(nomes.values())}
                     ).to_markdown(index=False),
        "\n\n## Perfil (medianas por cluster)\n\n",
        perfil[["pc_alfabetizado_2024", "pc_alfabetizado_2023", "populacao_2024",
                "pct_escolas_rurais", "pib_per_capita_2023", "idhm_e",
                "alunos_por_turma_ai"]].to_markdown(floatfmt=".2f"),
        "\n\n## Validação externa — taxa de atingimento da meta 2025 por cluster\n",
        "\n(não usada no fit; Brasil = 72,5%)\n\n",
        (taxa * 100).round(1).to_frame("atingiu_meta_2025_%").to_markdown(),
        "\n\n## Distribuição por região (contagem)\n\n",
        dist_regiao.to_markdown(),
        "\n",
    ]
    MD_PATH.write_text("".join(md), encoding="utf-8")
    log.info("gravado %s", MD_PATH)
    return out


if __name__ == "__main__":
    run()
