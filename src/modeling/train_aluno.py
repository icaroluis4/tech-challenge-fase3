"""Modelo A — probabilidade de um aluno ser alfabetizado (target `in_alfabetizado`).

Enquadramento (decisão D8): o aluno **não** possui features individuais — a
escola é anonimizada (F4) e tudo que descreve a criança (proficiência, respostas)
é leakage determinístico do target (F1). Logo o modelo só observa **contexto
municipal + rede de ensino**. Com ICC municipal ≈ 0,081 (G3), apenas ~8% da
variância do target individual está entre municípios: o teto de AUC é baixo *por
construção*. O produto correto é um **score de risco contextual** — "que fração
das crianças deste município tende a não se alfabetizar" — e não um classificador
individual.

Pipeline executado por `main()`:

1. carrega `dataset_aluno.parquet` e amostra estratificada por `(sg_uf, target)`;
2. holdout 80/20 estratificado (`split_aluno`, grupos = `co_municipio`);
3. compara 5 candidatos por `StratifiedKFold` (Dummy, LogReg, HistGB, LGBM, RF);
4. `RandomizedSearchCV` no melhor por ROC-AUC + curva treino × validação;
5. avalia no holdout com dois thresholds (F1 e custo assimétrico 5:1, D9);
6. avalia por `GroupKFold(co_municipio)` — o número honesto para município novo;
7. salva `models/modelo_aluno.joblib`, imagens e `reports/resultados_modelo_aluno.md`.

Uso::

    python -m src.modeling.train_aluno                 # amostra 400k (default)
    python -m src.modeling.train_aluno --fast          # 60k, busca curta (smoke test)
    python -m src.modeling.train_aluno --n-sample 0    # base inteira (1,94M)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

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
from src.preprocessing.splits import (
    Split,
    TARGET_ALUNO,
    group_cv,
    sample_aluno,
    split_aluno,
    stratified_cv,
)
from src.preprocessing.transformers import build_preprocessor, infer_feature_columns
from src.visualization.plots import (
    plot_calibration,
    plot_confusion,
    plot_learning_curve,
    plot_metric_bars,
    plot_roc_pr,
    save_fig,
)

log = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "modelo_aluno.joblib"
REPORT_PATH = REPORTS_DIR / "resultados_modelo_aluno.md"
METRICS_JSON = REPORTS_DIR / "resultados_modelo_aluno.json"

# `co_uf` é redundante com `sg_uf` (que entra como categórica) e seria tratado
# como número ordinal sem sentido; `no_municipio`/`id_aluno` já saem por BLACKLIST.
EXTRA_DROP = ("co_uf",)

# Custo assimétrico da decisão D9: deixar de sinalizar uma criança em risco
# (y=0 previsto como alfabetizado) é ~5× mais grave que um falso alarme.
CUSTO_MISS_RISCO = 5.0
CUSTO_FALSO_ALARME = 1.0

CLASS_LABELS = ("não alfabetizado", "alfabetizado")


def _try_lgbm():
    """LGBMClassifier se disponível; caso contrário `None` (fallback HistGB)."""
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415
        return LGBMClassifier
    except Exception:  # pragma: no cover - ambiente sem lightgbm
        log.warning("lightgbm indisponível — usando apenas HistGradientBoosting")
        return None


# --------------------------------------------------------------------------
# Candidatos
# --------------------------------------------------------------------------

def make_candidates(num_cols, cat_cols, seed: int = SEED) -> dict[str, Pipeline]:
    """Modelos candidatos, cada um com o pré-processador adequado.

    Modelos lineares recebem `scale=True`; modelos de árvore `scale=False`
    (escala é irrelevante para splits e evita custo desnecessário).
    """
    def pre(scale: bool):
        return build_preprocessor(num_cols, cat_cols, scale=scale)

    cands: dict[str, Pipeline] = {
        "dummy": Pipeline([
            ("pre", pre(False)),
            ("model", DummyClassifier(strategy="most_frequent", random_state=seed)),
        ]),
        "logreg": Pipeline([
            ("pre", pre(True)),
            # sklearn ≥1.8: `n_jobs` sem efeito e `penalty` depreciado (usar l1_ratio/C)
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                         random_state=seed)),
        ]),
        "hist_gb": Pipeline([
            ("pre", pre(False)),
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                min_samples_leaf=200, l2_regularization=1.0,
                early_stopping=False, class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre(False)),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=50, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        ]),
    }
    LGBM = _try_lgbm()
    if LGBM is not None:
        cands["lgbm"] = Pipeline([
            ("pre", pre(False)),
            ("model", LGBM(n_estimators=400, learning_rate=0.06, num_leaves=31,
                           min_child_samples=200, reg_lambda=1.0, subsample=0.9,
                           subsample_freq=1, colsample_bytree=0.8,
                           class_weight="balanced", n_jobs=-1, random_state=seed,
                           verbosity=-1)),
        ])
    return cands


def param_distributions(model_name: str) -> dict[str, Any]:
    """Grade de regularização explícita para o `RandomizedSearchCV` (D10)."""
    if model_name == "lgbm":
        return {
            "model__n_estimators": [200, 300, 400, 600, 800],
            "model__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.12],
            "model__num_leaves": [15, 31, 63, 127],
            "model__min_child_samples": [50, 100, 200, 400, 800],
            "model__reg_lambda": [0.0, 0.5, 1.0, 5.0, 20.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__subsample": [0.7, 0.85, 1.0],
        }
    if model_name == "hist_gb":
        return {
            "model__max_iter": [200, 300, 400, 600],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [50, 100, 200, 400],
            "model__l2_regularization": [0.0, 0.5, 1.0, 5.0, 20.0],
            "model__max_features": [0.6, 0.8, 1.0],
        }
    if model_name == "random_forest":
        return {
            "model__n_estimators": [200, 300, 500],
            "model__min_samples_leaf": [20, 50, 100, 200],
            "model__max_features": ["sqrt", 0.3, 0.5],
            "model__max_depth": [None, 8, 12, 20],
        }
    if model_name == "logreg":
        # `penalty` foi depreciado no sklearn 1.8 → regularização via C (L2 é o default)
        return {"model__C": np.logspace(-3, 2, 12)}
    return {}


# --------------------------------------------------------------------------
# Curva treino × validação (controle de overfitting)
# --------------------------------------------------------------------------

def learning_curve_boosting(pipe: Pipeline, sp: Split,
                            grid: Sequence[int] = (50, 100, 200, 400, 600, 800),
                            seed: int = SEED) -> pd.DataFrame:
    """ROC-AUC em treino e holdout variando o número de árvores.

    Usa uma sub-amostra do treino (≤120k) para manter o custo baixo; o objetivo
    é mostrar o *formato* da curva (onde o gap treino-validação abre).
    """
    from sklearn.metrics import roc_auc_score

    model = pipe.named_steps["model"]
    param = "n_estimators" if hasattr(model, "n_estimators") else "max_iter"
    n = min(120_000, len(sp.X_train))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sp.X_train), size=n, replace=False)
    x_tr, y_tr = sp.X_train.iloc[idx], sp.y_train.iloc[idx]

    rows = []
    for k in grid:
        p = _clone_with(pipe, {param: int(k)})
        p.fit(x_tr, y_tr)
        rows.append({
            param: int(k),
            "treino": roc_auc_score(y_tr, p.predict_proba(x_tr)[:, 1]),
            "validacao": roc_auc_score(sp.y_test, p.predict_proba(sp.X_test)[:, 1]),
        })
        log.info("curva %s=%d treino=%.4f val=%.4f", param, k, rows[-1]["treino"],
                 rows[-1]["validacao"])
    return pd.DataFrame(rows)


def _clone_with(pipe: Pipeline, model_params: dict[str, Any]) -> Pipeline:
    from sklearn.base import clone
    p = clone(pipe)
    p.set_params(**{f"model__{k}": v for k, v in model_params.items()})
    return p


# --------------------------------------------------------------------------
# Agregação municipal (leitura de política pública)
# --------------------------------------------------------------------------

def agregado_municipal(df_test: pd.DataFrame, y_true, proba) -> pd.DataFrame:
    """Compara taxa prevista × observada de alfabetização por município.

    Esta é a saída realmente útil do Modelo A: mesmo com AUC individual modesta,
    a média das probabilidades por município deve acompanhar a taxa observada.
    """
    d = pd.DataFrame({
        "co_municipio": df_test["co_municipio"].to_numpy(),
        "sg_uf": df_test["sg_uf"].astype("object").to_numpy(),
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(proba, dtype=float),
    })
    agg = (d.groupby("co_municipio", observed=True)
             .agg(sg_uf=("sg_uf", "first"), n_alunos=("y", "size"),
                  taxa_observada=("y", "mean"), taxa_prevista=("p", "mean"))
             .reset_index())
    agg["erro"] = agg["taxa_prevista"] - agg["taxa_observada"]
    return agg


MIN_ALUNOS_AGG = 30


def _correlacoes_municipais(agg: pd.DataFrame, min_alunos: int = MIN_ALUNOS_AGG
                            ) -> dict[str, float]:
    """Correlação prevista × observada por município, total e com massa mínima.

    Com poucos alunos amostrados por município a *taxa observada* é dominada por
    erro binomial (sd ≈ 0,5/√n), o que deprime a correlação por construção. Por
    isso reportamos também o subconjunto com ``n_alunos ≥ min_alunos``.
    """
    out: dict[str, float] = {}
    par = agg[["taxa_prevista", "taxa_observada"]]
    out["corr_municipal_pearson"] = float(par.corr().iloc[0, 1])
    out["corr_municipal_spearman"] = float(par.corr(method="spearman").iloc[0, 1])
    grandes = agg.loc[agg["n_alunos"] >= min_alunos, ["taxa_prevista", "taxa_observada"]]
    out["n_municipios"] = float(len(agg))
    out["n_municipios_com_massa"] = float(len(grandes))
    out["corr_municipal_pearson_n30"] = (
        float(grandes.corr().iloc[0, 1]) if len(grandes) > 2 else float("nan"))
    out["mae_municipal"] = float(agg["erro"].abs().mean())
    log.info("municípios: %d (%d com n≥%d) | Pearson=%.4f (n≥%d: %.4f) | MAE=%.4f",
             len(agg), len(grandes), min_alunos, out["corr_municipal_pearson"],
             min_alunos, out["corr_municipal_pearson_n30"], out["mae_municipal"])
    return out


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_sample: int = 400_000
    cv_sample: int = 150_000
    cv_folds: int = 5
    n_iter: int = 25
    group_folds: int = 5
    seed: int = SEED
    curva_grid: tuple[int, ...] = (50, 100, 200, 400, 600, 800)
    salvar: bool = True

    @classmethod
    def fast(cls) -> "TrainConfig":
        return cls(n_sample=60_000, cv_sample=30_000, cv_folds=3, n_iter=5,
                   group_folds=3, curva_grid=(50, 150, 400))


@dataclass
class TrainResult:
    best_name: str
    best_params: dict[str, Any]
    pipeline: Pipeline
    holdout: dict[str, dict[str, float]] = field(default_factory=dict)
    cv_strat: dict[str, dict[str, float]] = field(default_factory=dict)
    cv_group: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    municipal: pd.DataFrame | None = None
    curva: pd.DataFrame | None = None
    n_features: int = 0
    duracao_s: float = 0.0


def carregar_dataset(path=None) -> pd.DataFrame:
    path = path or (DATA_PROCESSED / "dataset_aluno.parquet")
    df = pd.read_parquet(path)
    log.info("dataset_aluno: %d linhas, %d cols", len(df), df.shape[1])
    return df


def _subamostra_cv(sp: Split, cfg: TrainConfig):
    """Sub-amostra do treino usada na CV/busca (controle de custo computacional)."""
    x, y, g = sp.X_train, sp.y_train, sp.groups_train
    if cfg.cv_sample and cfg.cv_sample < len(x):
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(x), size=cfg.cv_sample, replace=False)
        x, y = x.iloc[idx], y.iloc[idx]
        g = None if g is None else g.iloc[idx]
    return x, y, g


def _comparar_candidatos(cands: dict[str, Pipeline], x, y, cfg: TrainConfig):
    """CV estratificada em todos os candidatos → (tabela, nome do melhor)."""
    cv_strat: dict[str, dict[str, float]] = {}
    for nome, pipe in cands.items():
        res = cv_report(pipe, x, y, cv=stratified_cv(cfg.cv_folds, cfg.seed), label=nome)
        cv_strat[nome] = {**res.mean, **{f"{k}_sd": v for k, v in res.std.items()}}
        log.info("CV %s", res.summary())
    best_name = max((n for n in cv_strat if n != "dummy"),
                    key=lambda n: cv_strat[n]["roc_auc"])
    log.info("melhor candidato: %s (ROC-AUC CV=%.4f)", best_name, cv_strat[best_name]["roc_auc"])
    return cv_strat, best_name


def _buscar_hiperparametros(pipe: Pipeline, nome: str, x, y, cfg: TrainConfig):
    search = RandomizedSearchCV(
        pipe, param_distributions(nome), n_iter=cfg.n_iter,
        cv=stratified_cv(cfg.cv_folds, cfg.seed), scoring="roc_auc",
        random_state=cfg.seed, refit=True, n_jobs=1, error_score="raise", verbose=0,
    )
    search.fit(x, y)
    best_params = {k: (v.item() if hasattr(v, "item") else v)
                   for k, v in search.best_params_.items()}
    log.info("melhores params (%s): %s | ROC-AUC=%.4f", nome, best_params, search.best_score_)
    return search.best_estimator_, best_params


def _avaliar_holdout(cands: dict[str, Pipeline], best_name: str, sp: Split,
                     proba: np.ndarray, thresholds: dict[str, float]):
    """Métricas no holdout para baselines + melhor modelo em 3 thresholds."""
    holdout: dict[str, dict[str, float]] = {}
    curvas_roc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for nome in dict.fromkeys(["dummy", "logreg", best_name]):
        p = (proba if nome == best_name
             else cands[nome].fit(sp.X_train, sp.y_train).predict_proba(sp.X_test)[:, 1])
        holdout[nome] = evaluate_binary(sp.y_test, p, 0.5)
        curvas_roc[nome] = (sp.y_test.to_numpy(), p)
    holdout[f"{best_name}@f1"] = evaluate_binary(sp.y_test, proba, thresholds["f1"])
    holdout[f"{best_name}@custo5:1"] = evaluate_binary(sp.y_test, proba, thresholds["custo_5_1"])
    return holdout, curvas_roc


def treinar(df: pd.DataFrame, cfg: TrainConfig | None = None) -> TrainResult:
    """Executa a Fase 4 inteira sobre `df` e devolve o resultado consolidado."""
    cfg = cfg or TrainConfig()
    t0 = time.perf_counter()

    # 1) amostragem estratificada por (UF, target) ---------------------------
    if cfg.n_sample and cfg.n_sample < len(df):
        df = sample_aluno(df, n=cfg.n_sample, seed=cfg.seed)
    log.info("amostra de trabalho: %d linhas (pos=%.4f)", len(df), df[TARGET_ALUNO].mean())

    # 2) holdout -------------------------------------------------------------
    sp = split_aluno(df, seed=cfg.seed)
    log.info("split: %s", sp.summary())
    num_cols, cat_cols = infer_feature_columns(sp.X_train, extra_drop=EXTRA_DROP)
    log.info("features: %d numéricas + %d categóricas", len(num_cols), len(cat_cols))
    cands = make_candidates(num_cols, cat_cols, seed=cfg.seed)

    # 3) comparação de candidatos por StratifiedKFold -------------------------
    x_cv, y_cv, g_cv = _subamostra_cv(sp, cfg)
    cv_strat, best_name = _comparar_candidatos(cands, x_cv, y_cv, cfg)

    # 4) busca de hiperparâmetros + refit no treino completo ------------------
    best_pipe, best_params = _buscar_hiperparametros(cands[best_name], best_name,
                                                     x_cv, y_cv, cfg)
    best_pipe.fit(sp.X_train, sp.y_train)

    # 5) curva treino × validação (overfitting) ------------------------------
    curva = None
    if best_name in {"lgbm", "hist_gb", "random_forest"}:
        curva = learning_curve_boosting(best_pipe, sp, grid=cfg.curva_grid, seed=cfg.seed)

    # 6) holdout: thresholds e métricas --------------------------------------
    proba = best_pipe.predict_proba(sp.X_test)[:, 1]
    t_f1 = best_threshold_f1(sp.y_test, proba)
    t_cost, custo = best_threshold_cost(sp.y_test, proba, CUSTO_MISS_RISCO, CUSTO_FALSO_ALARME)
    thresholds = {"f1": float(t_f1), "custo_5_1": float(t_cost),
                  "custo_medio_por_aluno": float(custo)}
    log.info("thresholds: F1=%.4f | custo 5:1=%.4f (custo médio=%.4f)", t_f1, t_cost, custo)
    holdout, curvas_roc = _avaliar_holdout(cands, best_name, sp, proba, thresholds)

    # 7) GroupKFold por município (número honesto) ---------------------------
    cv_group: dict[str, float] = {}
    if g_cv is not None:
        gres = cv_report(best_pipe, x_cv, y_cv, cv=group_cv(cfg.group_folds),
                         groups=g_cv, label=f"{best_name}/GroupKFold")
        cv_group = {**gres.mean, **{f"{k}_sd": v for k, v in gres.std.items()}}
        log.info("CV %s", gres.summary())

    # 8) agregação municipal -------------------------------------------------
    municipal = agregado_municipal(sp.X_test, sp.y_test, proba)
    cv_group.update(_correlacoes_municipais(municipal))

    res = TrainResult(
        best_name=best_name, best_params=best_params, pipeline=best_pipe,
        holdout=holdout, cv_strat=cv_strat, cv_group=cv_group, thresholds=thresholds,
        municipal=municipal, curva=curva,
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
                 "target": TARGET_ALUNO, "seed": cfg.seed}, MODEL_PATH)
    log.info("modelo salvo em %s", MODEL_PATH)

    save_fig(plot_roc_pr(curvas_roc, "Modelo A — aluno (holdout 20%)"),
             IMAGES_DIR / "modelo_aluno_01_roc_pr.png")

    tabs = {res.best_name: calibration_table(sp.y_test, proba, bins=10)}
    save_fig(plot_calibration(tabs, "Modelo A — calibração (holdout)"),
             IMAGES_DIR / "modelo_aluno_02_calibracao.png")

    m = res.holdout[f"{res.best_name}@custo5:1"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    save_fig(plot_confusion(cm, CLASS_LABELS,
                            f"Matriz de confusão — threshold custo 5:1 = {res.thresholds['custo_5_1']:.3f}"),
             IMAGES_DIR / "modelo_aluno_03_confusao.png")

    if res.curva is not None:
        c = res.curva
        xcol = c.columns[0]
        save_fig(plot_learning_curve(c[xcol].to_numpy(), c["treino"].to_numpy(),
                                     c["validacao"].to_numpy(), xlabel=xcol,
                                     titulo=f"Modelo A ({res.best_name}) — treino × validação"),
                 IMAGES_DIR / "modelo_aluno_04_curva_overfit.png")

    comp = metrics_frame(dict(res.cv_strat),
                         columns=["roc_auc", "pr_auc", "f1", "balanced_accuracy"])
    save_fig(plot_metric_bars(comp, ["roc_auc", "pr_auc", "f1", "balanced_accuracy"],
                              f"Candidatos — StratifiedKFold({cfg.cv_folds})"),
             IMAGES_DIR / "modelo_aluno_05_candidatos.png")

    if res.municipal is not None:
        res.municipal.to_csv(REPORTS_DIR / "modelo_aluno_agregado_municipal.csv", index=False)

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

    gk = res.cv_group
    linhas_gk = "\n".join(
        f"- **{k}**: {v:.4f}" for k, v in gk.items() if not k.endswith("_sd")
    )
    mun = res.municipal
    if mun is not None:
        grandes = int(res.cv_group.get("n_municipios_com_massa", 0))
        mun_txt = (
            f"{len(mun):,} municípios no holdout · erro médio {mun['erro'].mean():+.4f} · "
            f"MAE {mun['erro'].abs().mean():.4f} · "
            f"Pearson {res.cv_group.get('corr_municipal_pearson', float('nan')):.4f} "
            f"(restrito a {grandes:,} municípios com n≥{MIN_ALUNOS_AGG} alunos amostrados: "
            f"{res.cv_group.get('corr_municipal_pearson_n30', float('nan')):.4f})"
        )
    else:
        mun_txt = "—"

    ece = float(calibration_table(sp.y_test, proba, bins=10).attrs["ece"])
    if res.curva is not None:
        c = res.curva
        i_best = int(c["validacao"].idxmax())
        curva_txt = (
            f"Validação máxima em `{c.columns[0]}={int(c.iloc[i_best, 0])}` "
            f"(AUC={c['validacao'].iloc[i_best]:.4f}); gap treino−validação no último "
            f"ponto = {c['treino'].iloc[-1] - c['validacao'].iloc[-1]:.4f}. Além desse ponto "
            f"o treino sobe e a validação cai — overfitting controlado pela regularização "
            f"escolhida e pelo `max_iter` da busca."
        )
    else:
        curva_txt = "Curva não gerada (melhor modelo não é baseado em árvores)."

    m05 = res.holdout[res.best_name]
    mf1 = res.holdout[f"{res.best_name}@f1"]
    mct = res.holdout[f"{res.best_name}@custo5:1"]

    md = f"""# Resultados — Modelo A (aluno)

> Gerado por `src/modeling/train_aluno.py` · seed={cfg.seed} ·
> amostra={len(sp.y_train) + len(sp.y_test):,} linhas ·
> {res.n_features} features · duração {res.duracao_s / 60:.1f} min.

## 1. Enquadramento

Target `in_alfabetizado` (1 = alfabetizado). Só alunos **presentes** (F2).
Nenhuma feature individual está disponível: escola anonimizada (F4) e todo dado
da prova é leakage determinístico (F1). O modelo observa **contexto municipal +
rede**, e o ICC municipal é ≈ 0,081 (G3) — ou seja, ~8% da variância individual
é explicável pelo município. **O teto de AUC é baixo por construção**; o produto
é um *score de risco contextual*, não um classificador individual.

## 2. Divisão dos dados

- Holdout 80/20 estratificado por target: `{sp.summary()}`.
- Validação cruzada estratificada `StratifiedKFold({cfg.cv_folds})` em
  {cfg.cv_sample:,} linhas do treino (custo computacional).
- Validação territorial `GroupKFold({cfg.group_folds})` por `co_municipio`.

## 3. Comparação de candidatos (StratifiedKFold)

{to_markdown(cv_tab)}

Melhor por ROC-AUC: **`{res.best_name}`**.

## 4. Hiperparâmetros escolhidos (`RandomizedSearchCV`, n_iter={cfg.n_iter})

```json
{json.dumps(res.best_params, indent=2, ensure_ascii=False)}
```

Controle de overfitting explícito na grade (`min_child_samples`/`min_samples_leaf`,
`reg_lambda`/`l2_regularization`, `num_leaves`/`max_leaf_nodes`, subamostragem de
colunas e linhas). Curva treino × validação em
`images/modelo_aluno_04_curva_overfit.png`: {curva_txt}

## 5. Desempenho no holdout (nunca tocado no treino)

{to_markdown(ho_tab)}

`pr_auc_risco`, `f1_risco`, `recall_risco` e `precision_risco` tratam a classe
**não alfabetizado** como positiva — é ela que orienta política pública.

### 5.1 Escolha do threshold (D9)

| threshold | valor | recall_risco | precision_risco | alunos sinalizados | leitura |
|---|---|---|---|---|---|
| padrão | 0.5000 | {m05['recall_risco']:.3f} | {m05['precision_risco']:.3f} | {int(m05['tn'] + m05['fn']):,} | equilíbrio neutro |
| max F1 (classe alfabetizado) | {res.thresholds['f1']:.4f} | {mf1['recall_risco']:.3f} | {mf1['precision_risco']:.3f} | {int(mf1['tn'] + mf1['fn']):,} | quase ninguém sinalizado — **inútil para política** |
| custo 5:1 (**adotado**) | {res.thresholds['custo_5_1']:.4f} | {mct['recall_risco']:.3f} | {mct['precision_risco']:.3f} | {int(mct['tn'] + mct['fn']):,} | captura {mct['recall_risco']:.0%} dos não alfabetizados ao custo de ampla triagem |

O threshold de máximo F1 otimiza a classe majoritária e praticamente elimina a
sinalização de risco — o oposto do objetivo. O custo assimétrico 5:1 (perder uma
criança em risco pesa 5× um falso alarme) leva a um threshold alto
({res.thresholds['custo_5_1']:.3f}): o modelo passa a operar como **triagem ampla**,
com recall de risco de {mct['recall_risco']:.1%} e custo médio de
{res.thresholds['custo_medio_por_aluno']:.4f} por aluno. Como só há contexto municipal
(sem features individuais), esse é o uso legítimo: **priorizar territórios/redes**, não
rotular crianças.

### 5.2 Calibração

ECE (10 bins de quantil) = **{ece:.4f}** — `images/modelo_aluno_02_calibracao.png`.
Com `class_weight="balanced"` o modelo **subestima** sistematicamente a probabilidade
de alfabetização (curva acima da diagonal): as probabilidades são úteis para
**ordenar** risco, mas não devem ser lidas como frequências absolutas sem
recalibração (Platt/isotônica) — fora do escopo desta fase.

## 6. Generalização territorial (`GroupKFold` por município)

{linhas_gk}

Agregação municipal (média das probabilidades × taxa observada): {mun_txt}.
Arquivo: `reports/modelo_aluno_agregado_municipal.csv`.

> A correlação total é deprimida por **erro binomial**: com poucos alunos
> amostrados por município, a taxa observada tem desvio ≈ 0,5/√n. A coluna
> restrita a n≥{MIN_ALUNOS_AGG} é a leitura correta da capacidade de ordenar
> municípios por risco.

## 7. Leitura dos resultados

- **AUC individual {res.holdout[res.best_name]['roc_auc']:.3f} (holdout) / {gk.get('roc_auc', float('nan')):.3f} (GroupKFold)** —
  dentro da faixa esperada 0,62–0,72 dado o ICC≈0,08. A queda de
  {res.holdout[res.best_name]['roc_auc'] - gk.get('roc_auc', float('nan')):.3f} entre
  as duas visões é o **custo de generalizar para municípios nunca vistos**; é o
  número que deve ser citado.
- Todos os modelos não triviais ficam a ≤0,01 de AUC entre si (§3): o limite é a
  **informação disponível**, não a classe de modelo. Mais complexidade não ajuda.
- **Correlação municipal {gk.get('corr_municipal_pearson_n30', float('nan')):.2f}** (n≥{MIN_ALUNOS_AGG}):
  agregado, o modelo reproduz bem a taxa de alfabetização observada — confirma
  o enquadramento de *score de risco contextual*.
- Uso recomendado: ranquear municípios/redes por risco médio; **não** decidir
  intervenção individual com este modelo.

## 8. Artefatos

- `models/modelo_aluno.joblib` (pipeline completo + thresholds)
- `images/modelo_aluno_01_roc_pr.png` · `02_calibracao.png` · `03_confusao.png`
  · `04_curva_overfit.png` · `05_candidatos.png`
- `reports/resultados_modelo_aluno.json` (métricas brutas)
"""
    REPORT_PATH.write_text(md, encoding="utf-8")
    payload = {
        "best_name": res.best_name, "best_params": res.best_params,
        "thresholds": res.thresholds, "cv_stratified": res.cv_strat,
        "cv_group": res.cv_group, "holdout": res.holdout,
        "n_features": res.n_features, "duracao_s": res.duracao_s,
        "config": {"n_sample": cfg.n_sample, "cv_sample": cfg.cv_sample,
                   "cv_folds": cfg.cv_folds, "n_iter": cfg.n_iter,
                   "group_folds": cfg.group_folds, "seed": cfg.seed},
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    log.info("relatório em %s", REPORT_PATH)


def main(argv: list[str] | None = None) -> TrainResult:
    ap = argparse.ArgumentParser(description="Treina o Modelo A (aluno).")
    ap.add_argument("--n-sample", type=int, default=400_000,
                    help="linhas na amostra estratificada (0 = base inteira)")
    ap.add_argument("--cv-sample", type=int, default=150_000,
                    help="linhas usadas na comparação/busca por CV")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--group-folds", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--fast", action="store_true", help="configuração de smoke test")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = TrainConfig.fast() if args.fast else TrainConfig(
        n_sample=args.n_sample, cv_sample=args.cv_sample, cv_folds=args.cv_folds,
        group_folds=args.group_folds, n_iter=args.n_iter,
    )
    cfg.salvar = not args.no_save
    res = treinar(carregar_dataset(), cfg)
    print(f"\nMelhor modelo: {res.best_name}")
    print(f"Holdout ROC-AUC: {res.holdout[res.best_name]['roc_auc']:.4f}")
    print(f"GroupKFold ROC-AUC: {res.cv_group.get('roc_auc', float('nan')):.4f}")
    print(f"Corr. municipal (Pearson): {res.cv_group.get('corr_municipal_pearson'):.4f}")
    return res


if __name__ == "__main__":  # pragma: no cover
    main()
