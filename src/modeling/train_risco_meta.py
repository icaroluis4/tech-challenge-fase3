"""Modelo B — risco de um município **não atingir** a meta de alfabetização.

Formulação genérica em ``t`` (ver PLANO_IMPLEMENTACAO.md §Fase 5): as features
de resultado/meta são renomeadas para um esquema relativo ao ano-alvo, de modo
que o **mesmo** pipeline treina com ``t=2025`` (backtest) e projeta ``t=2026``:

| genérico       | treino (t=2025)            | projeção (t=2026)                  |
|----------------|----------------------------|------------------------------------|
| `resultado_t1` | `pc_alfabetizado_2024`     | `pc_alfabetizado_aeeb_2025`        |
| `resultado_t2` | `pc_alfabetizado_2023`     | `pc_alfabetizado_2024`             |
| `delta_t1`     | `pc_2024 - pc_2023`        | `pc_2025 - pc_2024`                |
| `meta_t`       | `meta_2025`                | `meta_2026`                        |
| `esforco_t`    | `meta_2025 - pc_2024`      | `meta_2026 - pc_2025`              |
| `atingiu_t1`   | `pc_2024 >= meta_2024`     | `atingiu_meta_2025`                |
| target         | `atingiu_meta_2025`        | — (predizer)                       |

As demais features (Censo, IBGE, ADH, território) são estáticas e entram idênticas
nos dois anos. Colunas de ano explícito (`pc_alfabetizado_2023/2024`,
`meta_2024..2030`, `delta_2023_2024`, `gap_meta_2024`, `esforco_2025`,
`ambicao_2030`, `atingiu_meta_2024`) são **dropadas** para não duplicar
informação nem vazar 2025 — conferido por `assert_no_leakage`.

A **ablação** (D13) treina o mesmo melhor modelo sem o bloco de histórico
(`resultado_t1/t2`, `delta_t1`, `meta_t`, `esforco_t`, `atingiu_t1`) para medir
o valor incremental do contexto socioeconômico sobre a inércia do resultado.

Uso::

    python -m src.modeling.train_risco_meta            # completo (rápido, ~5.4k linhas)
    python -m src.modeling.train_risco_meta --fast     # smoke (busca curta)
    python -m src.modeling.train_risco_meta --no-save  # não grava artefatos
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED, IMAGES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from src.evaluation.metrics import (
    best_threshold_cost,
    best_threshold_f1,
    calibration_table,
    cv_report,
    evaluate_binary,
    metrics_frame,
    to_markdown,
)
from src.preprocessing.leakage import assert_no_leakage
from src.preprocessing.splits import (
    TARGET_MUNICIPIO,
    Split,
    split_municipio,
    stratified_cv,
)
from src.preprocessing.transformers import build_preprocessor, infer_feature_columns
from src.visualization.plots import (
    plot_calibration,
    plot_confusion,
    plot_metric_bars,
    plot_roc_pr,
    save_fig,
)

log = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "modelo_risco_meta.joblib"
REPORT_PATH = REPORTS_DIR / "resultados_modelo_risco_meta.md"
METRICS_JSON = REPORTS_DIR / "resultados_modelo_risco_meta.json"

# Bloco de histórico/metas — removido na ablação "só contexto" (D13).
HISTORICO_COLS = ("resultado_t1", "resultado_t2", "delta_t1",
                  "meta_t", "esforco_t", "atingiu_t1")

# `co_uf` é redundante com `sg_uf` (categórica); ids/nomes saem por BLACKLIST.
EXTRA_DROP = ("co_uf",)

# Custo assimétrico: deixar de sinalizar um município que vai falhar a meta
# (y=0 previsto como atingiu) é ~5× mais grave que um falso alarme — mesma
# lógica da D9 do Modelo A.
CUSTO_MISS_RISCO = 5.0
CUSTO_FALSO_ALARME = 1.0

CLASS_LABELS = ("não atingiu", "atingiu")


def _try_lgbm():
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415
        return LGBMClassifier
    except Exception:  # pragma: no cover
        log.warning("lightgbm indisponível — usando apenas HistGradientBoosting")
        return None


# --------------------------------------------------------------------------
# Matriz genérica em t
# --------------------------------------------------------------------------

def montar_matriz_t(features: pd.DataFrame, targets: pd.DataFrame, t: int) -> pd.DataFrame:
    """Monta a matriz de features no esquema genérico para o ano-alvo ``t``.

    ``t=2025`` → treino/backtest (inclui a coluna-target `atingiu_meta_2025`);
    ``t=2026`` → projeção (sem target). Devolve uma linha por município.
    """
    if t not in (2025, 2026):
        raise ValueError(f"ano-alvo não suportado: {t}")

    f = features.copy()
    tgt = targets.copy()

    if t == 2025:
        res_t1, res_t2, meta_t = f["pc_alfabetizado_2024"], f["pc_alfabetizado_2023"], f["meta_2025"]
        atingiu_t1 = (f["pc_alfabetizado_2024"] >= f["meta_2024"]).astype("Int64")
    else:  # 2026
        res_t1, res_t2, meta_t = tgt["pc_alfabetizado_aeeb_2025"], f["pc_alfabetizado_2024"], f["meta_2026"]
        atingiu_t1 = tgt[TARGET_MUNICIPIO].astype("Int64")

    out = f.assign(
        resultado_t1=res_t1,
        resultado_t2=res_t2,
        delta_t1=res_t1 - res_t2,
        meta_t=meta_t,
        esforco_t=meta_t - res_t1,
        atingiu_t1=atingiu_t1,
    )

    # Colunas de ano explícito: duplicam o bloco genérico ou vazam 2025.
    # (`meta_t`/`esforco_t` já são genéricas — não podem cair no filtro.)
    genericas = set(HISTORICO_COLS)
    drop_cols = [c for c in out.columns if c not in genericas and (
        c.startswith("pc_alfabetizado_") or c.startswith("meta_")
        or c in {"delta_2023_2024", "gap_meta_2024", "esforco_2025",
                 "ambicao_2030", "atingiu_meta_2024"}
    )]
    out = out.drop(columns=drop_cols)

    if t == 2025:
        out = out.merge(tgt[["co_municipio", TARGET_MUNICIPIO]], on="co_municipio",
                        how="left", validate="1:1")

    # ids/nomes ficam na matriz apenas como metadado (join, ranking); nunca
    # entram em X — `infer_feature_columns` os remove via BLACKLIST.
    assert_no_leakage(out.columns,
                      allow=(["co_municipio", "no_municipio", TARGET_MUNICIPIO]
                             if t == 2025 else ["co_municipio", "no_municipio"]))
    return out


# --------------------------------------------------------------------------
# Candidatos (mesma família do Modelo A, sem class_weight — base 72/28)
# --------------------------------------------------------------------------

def make_candidates(num_cols, cat_cols, seed: int = SEED) -> dict[str, Pipeline]:
    def pre(scale: bool):
        return build_preprocessor(num_cols, cat_cols, scale=scale)

    cands: dict[str, Pipeline] = {
        "dummy": Pipeline([
            ("pre", pre(False)),
            ("model", DummyClassifier(strategy="most_frequent", random_state=seed)),
        ]),
        "logreg": Pipeline([
            ("pre", pre(True)),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "hist_gb": Pipeline([
            ("pre", pre(False)),
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=15,
                min_samples_leaf=50, l2_regularization=1.0,
                early_stopping=False, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre(False)),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                n_jobs=-1, random_state=seed)),
        ]),
    }
    LGBM = _try_lgbm()
    if LGBM is not None:
        cands["lgbm"] = Pipeline([
            ("pre", pre(False)),
            ("model", LGBM(n_estimators=400, learning_rate=0.06, num_leaves=15,
                           min_child_samples=50, reg_lambda=1.0, subsample=0.9,
                           subsample_freq=1, colsample_bytree=0.8,
                           n_jobs=-1, random_state=seed, verbosity=-1)),
        ])
    return cands


def param_distributions(model_name: str) -> dict[str, Any]:
    """Grade de regularização — base pequena (5,4k) → modelos rasos."""
    if model_name == "lgbm":
        return {
            "model__n_estimators": [100, 200, 300, 400, 600],
            "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
            "model__num_leaves": [7, 15, 31],
            "model__min_child_samples": [20, 50, 100, 200],
            "model__reg_lambda": [0.0, 0.5, 1.0, 5.0, 20.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__subsample": [0.7, 0.85, 1.0],
        }
    if model_name == "hist_gb":
        return {
            "model__max_iter": [100, 200, 300, 400],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [20, 50, 100, 200],
            "model__l2_regularization": [0.0, 0.5, 1.0, 5.0, 20.0],
            "model__max_features": [0.6, 0.8, 1.0],
        }
    if model_name == "random_forest":
        return {
            "model__n_estimators": [200, 300, 500],
            "model__min_samples_leaf": [10, 20, 50, 100],
            "model__max_features": ["sqrt", 0.3, 0.5],
            "model__max_depth": [None, 6, 10, 16],
        }
    if model_name == "logreg":
        return {"model__C": np.logspace(-3, 2, 12)}
    return {}


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    cv_folds: int = 5
    n_iter: int = 25
    seed: int = SEED
    salvar: bool = True

    @classmethod
    def fast(cls) -> "TrainConfig":
        return cls(cv_folds=3, n_iter=5)


@dataclass
class TrainResult:
    best_name: str
    best_params: dict[str, Any]
    pipeline: Pipeline
    holdout: dict[str, dict[str, float]] = field(default_factory=dict)
    cv_strat: dict[str, dict[str, float]] = field(default_factory=dict)
    ablacao: dict[str, dict[str, float]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    n_features: int = 0
    duracao_s: float = 0.0


def carregar_dados(features_path=None, targets_path=None):
    f = pd.read_parquet(features_path or (DATA_PROCESSED / "features_municipio.parquet"))
    t = pd.read_parquet(targets_path or (DATA_PROCESSED / "targets_municipio.parquet"))
    log.info("features_municipio: %d×%d | targets_municipio: %d×%d",
             *f.shape, *t.shape)
    return f, t


def _comparar_candidatos(cands: dict[str, Pipeline], x, y, cfg: TrainConfig):
    cv_strat: dict[str, dict[str, float]] = {}
    for nome, pipe in cands.items():
        res = cv_report(pipe, x, y, cv=stratified_cv(cfg.cv_folds, cfg.seed), label=nome)
        cv_strat[nome] = {**res.mean, **{f"{k}_sd": v for k, v in res.std.items()}}
        log.info("CV %s", res.summary())
    best_name = max((n for n in cv_strat if n != "dummy"),
                    key=lambda n: cv_strat[n]["roc_auc"])
    log.info("melhor candidato: %s (ROC-AUC CV=%.4f)", best_name,
             cv_strat[best_name]["roc_auc"])
    return cv_strat, best_name


def _buscar_hiperparametros(pipe: Pipeline, nome: str, x, y, cfg: TrainConfig):
    search = RandomizedSearchCV(
        pipe, param_distributions(nome), n_iter=cfg.n_iter,
        cv=stratified_cv(cfg.cv_folds, cfg.seed), scoring="roc_auc",
        random_state=cfg.seed, refit=True, n_jobs=1, error_score="raise",
    )
    search.fit(x, y)
    best_params = {k: (v.item() if hasattr(v, "item") else v)
                   for k, v in search.best_params_.items()}
    log.info("melhores params (%s): %s | ROC-AUC=%.4f", nome, best_params,
             search.best_score_)
    return search.best_estimator_, best_params


def _avaliar_holdout(cands: dict[str, Pipeline], best_name: str, sp: Split,
                     proba: np.ndarray, thresholds: dict[str, float]):
    holdout: dict[str, dict[str, float]] = {}
    curvas_roc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for nome in dict.fromkeys(["dummy", "logreg", best_name]):
        p = (proba if nome == best_name
             else cands[nome].fit(sp.X_train, sp.y_train).predict_proba(sp.X_test)[:, 1])
        holdout[nome] = evaluate_binary(sp.y_test, p, 0.5)
        curvas_roc[nome] = (sp.y_test.to_numpy(), p)
    holdout[f"{best_name}@f1"] = evaluate_binary(sp.y_test, proba, thresholds["f1"])
    holdout[f"{best_name}@custo5:1"] = evaluate_binary(sp.y_test, proba,
                                                       thresholds["custo_5_1"])
    return holdout, curvas_roc


def _ablacao_sem_historico(best_name: str, best_params: dict[str, Any],
                           sp: Split, cfg: TrainConfig) -> dict[str, dict[str, float]]:
    """Treina o melhor modelo sem o bloco de histórico/metas (D13).

    Mede o valor incremental do contexto socioeconômico sobre a inércia do
    resultado passado — se a AUC cair pouco, o modelo é "fácil demais" e o
    socioeconômico não agrega.
    """
    drop = [c for c in HISTORICO_COLS if c in sp.X_train.columns]
    x_tr, x_te = sp.X_train.drop(columns=drop), sp.X_test.drop(columns=drop)
    num_cols, cat_cols = infer_feature_columns(x_tr, extra_drop=EXTRA_DROP)
    pipe = make_candidates(num_cols, cat_cols, seed=cfg.seed)[best_name]
    pipe.set_params(**best_params)
    pipe.fit(x_tr, sp.y_train)
    proba = pipe.predict_proba(x_te)[:, 1]
    res_cv = cv_report(pipe, x_tr, sp.y_train, cv=stratified_cv(cfg.cv_folds, cfg.seed),
                       label=f"{best_name}/sem-histórico")
    return {
        "cv": {**res_cv.mean, **{f"{k}_sd": v for k, v in res_cv.std.items()}},
        "holdout": evaluate_binary(sp.y_test, proba, 0.5),
    }


def treinar(df: pd.DataFrame, cfg: TrainConfig | None = None) -> TrainResult:
    """Executa a Fase 5 (backtest t=2025) sobre a matriz genérica ``df``."""
    cfg = cfg or TrainConfig()
    t0 = time.perf_counter()

    sp = split_municipio(df, seed=cfg.seed)
    log.info("split: %s", sp.summary())
    num_cols, cat_cols = infer_feature_columns(sp.X_train, extra_drop=EXTRA_DROP)
    log.info("features: %d numéricas + %d categóricas", len(num_cols), len(cat_cols))
    cands = make_candidates(num_cols, cat_cols, seed=cfg.seed)

    cv_strat, best_name = _comparar_candidatos(cands, sp.X_train, sp.y_train, cfg)
    best_pipe, best_params = _buscar_hiperparametros(cands[best_name], best_name,
                                                     sp.X_train, sp.y_train, cfg)
    best_pipe.fit(sp.X_train, sp.y_train)

    proba = best_pipe.predict_proba(sp.X_test)[:, 1]
    t_f1 = best_threshold_f1(sp.y_test, proba)
    t_cost, custo = best_threshold_cost(sp.y_test, proba, CUSTO_MISS_RISCO,
                                        CUSTO_FALSO_ALARME)
    thresholds = {"f1": float(t_f1), "custo_5_1": float(t_cost),
                  "custo_medio_por_municipio": float(custo)}
    log.info("thresholds: F1=%.4f | custo 5:1=%.4f (custo médio=%.4f)",
             t_f1, t_cost, custo)
    holdout, curvas_roc = _avaliar_holdout(cands, best_name, sp, proba, thresholds)

    ablacao = _ablacao_sem_historico(best_name, best_params, sp, cfg)
    log.info("ablação sem histórico: ROC-AUC holdout=%.4f (completo=%.4f)",
             ablacao["holdout"]["roc_auc"], holdout[best_name]["roc_auc"])

    res = TrainResult(
        best_name=best_name, best_params=best_params, pipeline=best_pipe,
        holdout=holdout, cv_strat=cv_strat, ablacao=ablacao, thresholds=thresholds,
        n_features=len(num_cols) + len(cat_cols),
        duracao_s=time.perf_counter() - t0,
    )
    if cfg.salvar:
        _salvar_artefatos(res, sp, proba, curvas_roc, cfg)
    return res


def _salvar_artefatos(res: TrainResult, sp: Split, proba: np.ndarray,
                      curvas_roc: dict, cfg: TrainConfig) -> None:
    joblib.dump({"pipeline": res.pipeline, "best_name": res.best_name,
                 "best_params": res.best_params, "thresholds": res.thresholds,
                 "target": TARGET_MUNICIPIO, "seed": cfg.seed}, MODEL_PATH)
    log.info("modelo salvo em %s", MODEL_PATH)

    save_fig(plot_roc_pr(curvas_roc, "Modelo B — risco de meta (holdout 20%)"),
             IMAGES_DIR / "modelo_meta_01_roc_pr.png")

    tabs = {res.best_name: calibration_table(sp.y_test, proba, bins=10)}
    save_fig(plot_calibration(tabs, "Modelo B — calibração (holdout)"),
             IMAGES_DIR / "modelo_meta_02_calibracao.png")

    m = res.holdout[f"{res.best_name}@custo5:1"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    save_fig(plot_confusion(cm, CLASS_LABELS,
                            f"Matriz de confusão — threshold custo 5:1 = {res.thresholds['custo_5_1']:.3f}"),
             IMAGES_DIR / "modelo_meta_03_confusao.png")

    comp = metrics_frame(dict(res.cv_strat),
                         columns=["roc_auc", "pr_auc", "f1", "balanced_accuracy"])
    save_fig(plot_metric_bars(comp, ["roc_auc", "pr_auc", "f1", "balanced_accuracy"],
                              f"Candidatos — StratifiedKFold({cfg.cv_folds})"),
             IMAGES_DIR / "modelo_meta_04_candidatos.png")

    _escrever_relatorio(res, sp, proba, cfg)


def _escrever_relatorio(res: TrainResult, sp: Split, proba: np.ndarray,
                        cfg: TrainConfig) -> None:
    cv_tab = metrics_frame(res.cv_strat,
                           columns=["roc_auc", "roc_auc_sd", "pr_auc", "f1",
                                    "balanced_accuracy", "brier"])
    cv_tab.index.name = "modelo"
    ho_tab = metrics_frame(res.holdout,
                           columns=["threshold", "roc_auc", "pr_auc", "pr_auc_risco",
                                    "f1", "f1_risco", "balanced_accuracy", "accuracy",
                                    "recall_risco", "precision_risco", "brier"])
    ho_tab.index.name = "modelo"

    ab = res.ablacao
    ab_tab = pd.DataFrame({
        "modelo": [f"{res.best_name} (completo)", f"{res.best_name} (sem histórico)"],
        "roc_auc_cv": [res.cv_strat[res.best_name]["roc_auc"], ab["cv"]["roc_auc"]],
        "roc_auc_holdout": [res.holdout[res.best_name]["roc_auc"],
                            ab["holdout"]["roc_auc"]],
        "pr_auc_risco_holdout": [res.holdout[res.best_name]["pr_auc_risco"],
                                 ab["holdout"]["pr_auc_risco"]],
        "brier_holdout": [res.holdout[res.best_name]["brier"], ab["holdout"]["brier"]],
    }).set_index("modelo")

    ece = float(calibration_table(sp.y_test, proba, bins=10).attrs["ece"])
    m05 = res.holdout[res.best_name]
    mf1 = res.holdout[f"{res.best_name}@f1"]
    mct = res.holdout[f"{res.best_name}@custo5:1"]

    md = f"""# Resultados — Modelo B (risco de não atingimento de meta)

> Gerado por `src/modeling/train_risco_meta.py` · seed={cfg.seed} ·
> {len(sp.y_train) + len(sp.y_test):,} municípios rotulados ·
> {res.n_features} features · duração {res.duracao_s:.0f} s.

## 1. Enquadramento

Target `atingiu_meta_2025` (1 = atingiu; prevalência {sp.y_train.mean():.1%}).
A classe de interesse é **não atingiu** (y=0) — métricas `*_risco`. Formulação
genérica em ``t``: o mesmo pipeline treina com t=2025 (backtest) e projeta
t=2026 (`predict_2026.py`). Colunas de ano explícito são dropadas e o bloco de
histórico é renomeado (`resultado_t1/t2`, `delta_t1`, `meta_t`, `esforco_t`,
`atingiu_t1`).

## 2. Divisão dos dados

- Holdout 80/20 estratificado: `{sp.summary()}` (83 municípios sem rótulo
  descartados — G5).
- Validação cruzada `StratifiedKFold({cfg.cv_folds})` no treino.

## 3. Comparação de candidatos (StratifiedKFold)

{to_markdown(cv_tab)}

Melhor por ROC-AUC: **`{res.best_name}`**.

## 4. Hiperparâmetros (`RandomizedSearchCV`, n_iter={cfg.n_iter})

```json
{json.dumps(res.best_params, indent=2, ensure_ascii=False)}
```

## 5. Desempenho no holdout (backtest 2025)

{to_markdown(ho_tab)}

### 5.1 Thresholds

| threshold | valor | recall_risco | precision_risco | municípios sinalizados | leitura |
|---|---|---|---|---|---|
| padrão | 0.5000 | {m05['recall_risco']:.3f} | {m05['precision_risco']:.3f} | {int(m05['tn'] + m05['fn']):,} | equilíbrio neutro |
| max F1 (classe atingiu) | {res.thresholds['f1']:.4f} | {mf1['recall_risco']:.3f} | {mf1['precision_risco']:.3f} | {int(mf1['tn'] + mf1['fn']):,} | otimiza a classe majoritária |
| custo 5:1 (**adotado**) | {res.thresholds['custo_5_1']:.4f} | {mct['recall_risco']:.3f} | {mct['precision_risco']:.3f} | {int(mct['tn'] + mct['fn']):,} | captura {mct['recall_risco']:.0%} dos que falhariam a meta |

### 5.2 Calibração

ECE (10 bins) = **{ece:.4f}** — `images/modelo_meta_02_calibracao.png`. Aqui as
probabilidades viram **ranking de prioridade** de gestão, então a calibração
importa mais que no Modelo A; se o ECE estiver alto, aplicar
`CalibratedClassifierCV` isotônico antes de citar probabilidades absolutas.

## 6. Ablação — valor incremental do contexto socioeconômico (D13)

{to_markdown(ab_tab)}

O modelo "sem histórico" remove `resultado_t1/t2`, `delta_t1`, `meta_t`,
`esforco_t` e `atingiu_t1` — só restam Censo, IBGE, ADH e território. A queda
de AUC quantifica quanto do desempenho é **inércia do resultado passado**;
o que sobra é o valor preditivo do contexto socioeconômico.

## 7. Artefatos

- `models/modelo_risco_meta.joblib` (pipeline + thresholds)
- `images/modelo_meta_01_roc_pr.png` · `02_calibracao.png` · `03_confusao.png`
  · `04_candidatos.png`
- `reports/resultados_modelo_risco_meta.json` (métricas brutas)
- Projeção 2026: `python -m src.modeling.predict_2026` →
  `reports/ranking_risco_2026.csv`
"""
    REPORT_PATH.write_text(md, encoding="utf-8")
    payload = {
        "best_name": res.best_name, "best_params": res.best_params,
        "thresholds": res.thresholds, "cv_stratified": res.cv_strat,
        "ablacao_sem_historico": res.ablacao, "holdout": res.holdout,
        "n_features": res.n_features, "duracao_s": res.duracao_s,
        "config": {"cv_folds": cfg.cv_folds, "n_iter": cfg.n_iter, "seed": cfg.seed},
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    log.info("relatório em %s", REPORT_PATH)


def main(argv: list[str] | None = None) -> TrainResult:
    ap = argparse.ArgumentParser(description="Treina o Modelo B (risco de meta).")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--fast", action="store_true", help="configuração de smoke test")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = TrainConfig.fast() if args.fast else TrainConfig(
        cv_folds=args.cv_folds, n_iter=args.n_iter)
    cfg.salvar = not args.no_save
    features, targets = carregar_dados()
    df = montar_matriz_t(features, targets, t=2025)
    res = treinar(df, cfg)
    print(f"\nMelhor modelo: {res.best_name}")
    print(f"Holdout ROC-AUC: {res.holdout[res.best_name]['roc_auc']:.4f}")
    print(f"Holdout PR-AUC risco: {res.holdout[res.best_name]['pr_auc_risco']:.4f}")
    print(f"Ablação sem histórico (holdout ROC-AUC): "
          f"{res.ablacao['holdout']['roc_auc']:.4f}")
    return res


if __name__ == "__main__":  # pragma: no cover
    main()
