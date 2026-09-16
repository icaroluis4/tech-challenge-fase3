"""Projeção 2026 do Modelo B — ranking de risco de não atingimento de meta.

Monta a matriz genérica com ``t=2026`` (`montar_matriz_t`), aplica o pipeline
treinado em ``models/modelo_risco_meta.joblib`` e grava
``reports/ranking_risco_2026.csv`` ordenado por `p_nao_atingir_2026` desc.

Validações de sanidade (registradas no log e no rodapé do CSV não — no stdout):
- Spearman(`p_nao_atingir_2026`, `esforco_2026`) deve ser positiva e forte;
- municípios que não atingiram 2025 devem ter risco médio 2026 maior.

Uso::

    python -m src.modeling.predict_2026
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.modeling.train_risco_meta import (
    EXTRA_DROP,
    MODEL_PATH,
    carregar_dados,
    montar_matriz_t,
)
from src.preprocessing.transformers import infer_feature_columns

log = logging.getLogger(__name__)

RANKING_PATH = REPORTS_DIR / "ranking_risco_2026.csv"


def projetar_2026(model_path=None) -> pd.DataFrame:
    """Devolve o ranking de risco 2026 (uma linha por município)."""
    artefato = joblib.load(model_path or MODEL_PATH)
    pipe = artefato["pipeline"]
    log.info("modelo carregado: %s (%s)", model_path or MODEL_PATH,
             artefato.get("best_name"))

    features, targets = carregar_dados()
    df = montar_matriz_t(features, targets, t=2026)
    log.info("matriz t=2026: %d linhas, %d cols", len(df), df.shape[1])

    num_cols, cat_cols = infer_feature_columns(df, extra_drop=EXTRA_DROP)
    X = df[num_cols + cat_cols]
    p_atingir = pipe.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "co_municipio": df["co_municipio"].to_numpy(),
        "no_municipio": df["no_municipio"].astype("object").to_numpy(),
        "sg_uf": df["sg_uf"].astype("object").to_numpy(),
        "p_nao_atingir_2026": 1.0 - p_atingir,
        "meta_2026": df["meta_t"].to_numpy(),
        "pc_2025": df["resultado_t1"].to_numpy(),
        "esforco_2026": df["esforco_t"].to_numpy(),
        "atingiu_meta_2025": df["atingiu_t1"].astype("Int64").to_numpy(),
    }).sort_values("p_nao_atingir_2026", ascending=False).reset_index(drop=True)
    return out


def _sanidade(ranking: pd.DataFrame) -> dict[str, float]:
    """Checagens de coerência da projeção (D13)."""
    r = ranking.dropna(subset=["p_nao_atingir_2026", "esforco_2026"])
    spearman = float(r["p_nao_atingir_2026"].corr(r["esforco_2026"],
                                                  method="spearman"))
    rotulados = ranking.dropna(subset=["atingiu_meta_2025"])
    risco_falhou_2025 = float(
        rotulados.loc[rotulados["atingiu_meta_2025"] == 0, "p_nao_atingir_2026"].mean())
    risco_atingiu_2025 = float(
        rotulados.loc[rotulados["atingiu_meta_2025"] == 1, "p_nao_atingir_2026"].mean())
    log.info("sanidade: Spearman(risco, esforço)=%.3f | risco médio 2026: "
             "falhou 2025=%.3f vs atingiu 2025=%.3f",
             spearman, risco_falhou_2025, risco_atingiu_2025)
    return {
        "spearman_risco_esforco": spearman,
        "risco_medio_falhou_2025": risco_falhou_2025,
        "risco_medio_atingiu_2025": risco_atingiu_2025,
    }


def main() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ranking = projetar_2026()
    san = _sanidade(ranking)
    ranking.to_csv(RANKING_PATH, index=False)
    log.info("ranking salvo em %s (%d municípios)", RANKING_PATH, len(ranking))

    print(f"\nRanking 2026: {len(ranking):,} municípios -> {RANKING_PATH}")
    print(f"Spearman(risco, esforço) = {san['spearman_risco_esforco']:.3f}")
    print(f"Risco médio | falhou 2025: {san['risco_medio_falhou_2025']:.3f} · "
          f"atingiu 2025: {san['risco_medio_atingiu_2025']:.3f}")
    print("\nTop-10 risco de não atingir a meta 2026:")
    cols = ["no_municipio", "sg_uf", "p_nao_atingir_2026", "meta_2026",
            "pc_2025", "esforco_2026"]
    print(ranking[cols].head(10).to_string(index=False))
    return ranking


if __name__ == "__main__":  # pragma: no cover
    main()
