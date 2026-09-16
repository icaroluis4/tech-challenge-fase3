"""Fase 7 (light, D14) — Interpretabilidade dos Modelos A e B.

- **Permutation importance** (no holdout) para os dois modelos — responde às
  perguntas "quais fatores mais impactam" e "quais variáveis têm maior influência".
- **SHAP apenas no Modelo B** (logreg, 5,4k linhas — `LinearExplainer` é
  instantâneo e cobre o requisito nominal de SHAP). SHAP do Modelo A (HistGB,
  400k linhas) fica como evolução futura (D14).

Saídas:
- ``reports/interpretabilidade.md`` — tabelas top-15 (A e B) + leitura SHAP (B)
- ``images/interp_01_permutation_aluno.png``
- ``images/interp_02_permutation_risco.png``
- ``images/interp_03_shap_risco.png`` (beeswarm, Modelo B)

Uso::

    python -m src.evaluation.interpret
"""
from __future__ import annotations

import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import IMAGES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from src.modeling.train_aluno import carregar_dataset
from src.modeling.train_risco_meta import carregar_dados, montar_matriz_t
from src.preprocessing.splits import (
    TARGET_ALUNO,
    TARGET_MUNICIPIO,
    sample_aluno,
    split_aluno,
    split_municipio,
)
from src.preprocessing.transformers import get_feature_names

log = logging.getLogger(__name__)

TOP_N = 15
N_SAMPLE_ALUNO = 20_000  # permutation no Modelo A é cara — amostra do holdout

MD_PATH = REPORTS_DIR / "interpretabilidade.md"


def _permutation(pipe, x, y, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance (queda de ROC-AUC) agregada por feature original."""
    r = permutation_importance(pipe, x, y, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=SEED, n_jobs=1)
    return (pd.DataFrame({"feature": x.columns,
                          "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def _fig_importance(imp: pd.DataFrame, titulo: str, path, top: int = TOP_N):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="C0")
    ax.set_xlabel("queda de ROC-AUC (permutation)")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _shap_logreg(pipe, x: pd.DataFrame):
    """SHAP LinearExplainer no Modelo B (logreg) — valores na escala do logit."""
    import shap

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    xt = pre.transform(x)
    names = get_feature_names(pre)
    explainer = shap.LinearExplainer(model, xt, feature_names=names)
    shap_values = explainer.shap_values(xt)
    mean_abs = np.abs(shap_values).mean(axis=0)
    agg = (pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    return shap_values, names, xt, agg


def _fig_shap_beeswarm(shap_values, xt, names, path):
    import shap

    plt.figure()
    shap.summary_plot(shap_values, xt, feature_names=names, max_display=TOP_N,
                      show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    md: list[str] = ["# Interpretabilidade — Fase 7 (escopo reduzido, D14)\n",
                     "\nPermutation importance (queda de ROC-AUC no holdout) nos dois "
                     "modelos + SHAP `LinearExplainer` no Modelo B. SHAP do Modelo A "
                     "(HistGB, 400k) fica como evolução futura (D14).\n"]

    # ---------------- Modelo B (rápido — faz primeiro) ----------------
    log.info("=== Modelo B (risco de meta) ===")
    pipe_b = joblib.load(MODELS_DIR / "modelo_risco_meta.joblib")["pipeline"]
    features, targets = carregar_dados()
    mat = montar_matriz_t(features, targets, 2025)
    sp_b = split_municipio(mat)
    imp_b = _permutation(pipe_b, sp_b.X_test, sp_b.y_test)
    log.info("Modelo B top-5: %s", imp_b.head(5)["feature"].tolist())
    _fig_importance(imp_b, "Modelo B — permutation importance (holdout)",
                    IMAGES_DIR / "interp_02_permutation_risco.png")

    shap_values, names, xt, agg_b = _shap_logreg(pipe_b, sp_b.X_test)
    _fig_shap_beeswarm(shap_values, xt, names, IMAGES_DIR / "interp_03_shap_risco.png")
    log.info("SHAP Modelo B top-5: %s", agg_b.head(5)["feature"].tolist())

    md += ["\n## Modelo B — risco de não atingir a meta (logreg)\n",
           "\n### Permutation importance (top-15)\n\n",
           imp_b.head(TOP_N).to_markdown(index=False, floatfmt=".4f"),
           "\n\n### SHAP — |valor| médio (top-15, features já transformadas)\n\n",
           agg_b.head(TOP_N).to_markdown(index=False, floatfmt=".4f"),
           "\n\nBeeswarm: `images/interp_03_shap_risco.png`. "
           "Leitura: valores SHAP positivos empurram para **atingir** a meta "
           "(classe 1); negativos, para **não atingir**.\n"]

    # ---------------- Modelo A (amostra do holdout) ----------------
    log.info("=== Modelo A (aluno) ===")
    pipe_a = joblib.load(MODELS_DIR / "modelo_aluno.joblib")["pipeline"]
    df_a = sample_aluno(carregar_dataset(), n=400_000)
    sp_a = split_aluno(df_a)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(sp_a.X_test), size=min(N_SAMPLE_ALUNO, len(sp_a.X_test)),
                     replace=False)
    x_a, y_a = sp_a.X_test.iloc[idx], sp_a.y_test.iloc[idx]
    imp_a = _permutation(pipe_a, x_a, y_a, n_repeats=3)
    log.info("Modelo A top-5: %s", imp_a.head(5)["feature"].tolist())
    _fig_importance(imp_a, "Modelo A — permutation importance (holdout, 20k)",
                    IMAGES_DIR / "interp_01_permutation_aluno.png")

    md += ["\n## Modelo A — aluno alfabetizado (HistGB, amostra 20k do holdout)\n",
           "\n### Permutation importance (top-15)\n\n",
           imp_a.head(TOP_N).to_markdown(index=False, floatfmt=".4f"),
           "\n\nFigura: `images/interp_01_permutation_aluno.png`.\n",
           "\n## Leitura cruzada (A × B)\n",
           "\nAs duas listas devem ser dominadas por resultado histórico do "
           "município (`pc_alfabetizado_*`), rede/estrutura escolar e "
           "socioeconomia (IDHM/PIB) — coerente com ICC≈0,08 (risco contextual) "
           "e com a ablação D13 (contexto sozinho prediz, AUC 0,79).\n"]

    MD_PATH.write_text("".join(md), encoding="utf-8")
    log.info("gravado %s", MD_PATH)


if __name__ == "__main__":
    run()
