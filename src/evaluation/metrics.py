"""Métricas de classificação binária, calibração e relatórios de validação cruzada.

Convenção de classes usada em todo o projeto (Modelo A — aluno):

- ``y = 1`` → aluno **alfabetizado** (classe positiva, 66,2% da base);
- ``y = 0`` → aluno **não alfabetizado** = *criança em risco* (classe de interesse
  para política pública).

Por isso o módulo reporta métricas nas **duas** direções: `pr_auc` (positiva =
alfabetizado) e `pr_auc_risco` (positiva = não alfabetizado). Quem prioriza
intervenção olha `pr_auc_risco` e `recall_risco`.

O custo assimétrico também é expresso nessa linguagem:

- ``miss_risco`` → aluno em risco (y=0) classificado como alfabetizado (ŷ=1):
  a criança não é sinalizada → **erro caro**;
- ``falso_alarme`` → aluno alfabetizado (y=1) classificado como em risco (ŷ=0):
  gasta recurso de reforço em quem não precisava → erro barato.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

# Métricas usadas nos relatórios de CV (nomes do sklearn).
CV_SCORING: dict[str, str] = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "neg_brier": "neg_brier_score",
}

# Ordem canônica das colunas nas tabelas de resultado.
METRIC_ORDER: list[str] = [
    "roc_auc", "pr_auc", "pr_auc_risco", "f1", "f1_risco",
    "balanced_accuracy", "accuracy", "precision", "recall",
    "precision_risco", "recall_risco", "brier", "log_loss",
]


def _as_arrays(y_true, proba) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y_true).astype("float64"), dtype="float64")
    p = np.asarray(pd.Series(proba).astype("float64"), dtype="float64")
    if y.shape != p.shape:
        raise ValueError(f"y_true {y.shape} e proba {p.shape} têm shapes diferentes")
    if np.isnan(y).any() or np.isnan(p).any():
        raise ValueError("y_true/proba não podem conter NaN")
    if p.min() < 0 or p.max() > 1:
        raise ValueError("proba deve estar em [0, 1]")
    return y.astype(int), p


def evaluate_binary(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Dicionário completo de métricas para um vetor de probabilidades.

    Inclui as métricas "espelhadas" na classe de risco (y=0) e a matriz de
    confusão desempacotada (`tn`, `fp`, `fn`, `tp`), além de `miss_risco` e
    `falso_alarme` na linguagem de negócio.
    """
    y, p = _as_arrays(y_true, proba)
    pred = (p >= threshold).astype(int)

    single_class = len(np.unique(y)) < 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    out: dict[str, float] = {
        "threshold": float(threshold),
        "n": int(len(y)),
        "prevalencia": float(y.mean()),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y, p)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y, p)),
        # positiva = risco (y=0) → inverte rótulo e probabilidade
        "pr_auc_risco": float("nan") if single_class else float(average_precision_score(1 - y, 1 - p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f1_risco": float(f1_score(1 - y, 1 - pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision_risco": float(precision_score(1 - y, 1 - pred, zero_division=0)),
        "recall_risco": float(recall_score(1 - y, 1 - pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # linguagem de negócio (ver docstring do módulo)
        "miss_risco": int(fp),      # y=0 previsto como 1
        "falso_alarme": int(fn),    # y=1 previsto como 0
    }
    return out


def best_threshold_f1(y_true, proba, n_grid: int = 199, target: str = "positiva") -> float:
    """Threshold que maximiza F1 da classe `positiva` (alfabetizado) ou `risco`.

    Varre um grid de quantis da própria distribuição de probabilidades (mais
    estável que grid uniforme quando as probas ficam concentradas).
    """
    y, p = _as_arrays(y_true, proba)
    qs = np.linspace(0.005, 0.995, n_grid)
    grid = np.unique(np.quantile(p, qs))
    best_t, best_s = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        s = (f1_score(y, pred, zero_division=0) if target == "positiva"
             else f1_score(1 - y, 1 - pred, zero_division=0))
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t


def best_threshold_cost(
    y_true,
    proba,
    cost_miss_risco: float = 5.0,
    cost_falso_alarme: float = 1.0,
    n_grid: int = 199,
) -> tuple[float, float]:
    """Threshold que minimiza o custo esperado assimétrico.

    ``custo = cost_miss_risco · #(y=0, ŷ=1) + cost_falso_alarme · #(y=1, ŷ=0)``

    O default 5:1 reflete a decisão D9: deixar de sinalizar uma criança em risco
    é ~5× mais grave que oferecer reforço a quem já está alfabetizado.

    Retorna ``(threshold, custo_medio_por_aluno)``.
    """
    if cost_miss_risco <= 0 or cost_falso_alarme <= 0:
        raise ValueError("custos devem ser positivos")
    y, p = _as_arrays(y_true, proba)
    qs = np.linspace(0.005, 0.995, n_grid)
    grid = np.unique(np.quantile(p, qs))
    best_t, best_c = 0.5, float("inf")
    for t in grid:
        pred = (p >= t).astype(int)
        miss = int(((y == 0) & (pred == 1)).sum())
        alarme = int(((y == 1) & (pred == 0)).sum())
        c = (cost_miss_risco * miss + cost_falso_alarme * alarme) / len(y)
        if c < best_c:
            best_t, best_c = float(t), float(c)
    return best_t, best_c


def calibration_table(y_true, proba, bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Tabela de calibração: probabilidade média prevista × frequência observada.

    `strategy="quantile"` usa bins de tamanho aproximadamente igual (robusto
    quando as probabilidades são concentradas); `"uniform"` usa faixas fixas.
    """
    y, p = _as_arrays(y_true, proba)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        raise ValueError("strategy deve ser 'quantile' ou 'uniform'")
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    tab = (
        df.groupby("bin", observed=True)
          .agg(n=("y", "size"), p_media=("p", "mean"), freq_observada=("y", "mean"),
               p_min=("p", "min"), p_max=("p", "max"))
          .reset_index()
    )
    tab["erro"] = tab["freq_observada"] - tab["p_media"]
    # Expected Calibration Error (ponderado pelo tamanho do bin)
    tab.attrs["ece"] = float(np.average(tab["erro"].abs(), weights=tab["n"]))
    return tab


def expected_calibration_error(y_true, proba, bins: int = 10) -> float:
    """ECE em bins de quantil — quanto menor, melhor calibrada a probabilidade."""
    return float(calibration_table(y_true, proba, bins=bins).attrs["ece"])


@dataclass
class CVResult:
    """Resultado de `cv_report`: médias, desvios e folds brutos."""
    mean: dict[str, float]
    std: dict[str, float]
    folds: pd.DataFrame
    label: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"metrica": list(self.mean), "media": list(self.mean.values()),
                             "desvio": [self.std[k] for k in self.mean]})

    def summary(self) -> str:
        parts = [f"{k}={self.mean[k]:.4f}±{self.std[k]:.4f}" for k in self.mean]
        prefix = f"[{self.label}] " if self.label else ""
        return prefix + " | ".join(parts)


def cv_report(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None = None,
    scoring: Mapping[str, str] | None = None,
    n_jobs: int | None = None,
    label: str = "",
) -> CVResult:
    """Validação cruzada com múltiplas métricas → média ± desvio por métrica.

    `groups` não-nulo + `cv=GroupKFold` dá o número honesto de generalização
    territorial (município nunca visto no treino).
    """
    scoring = dict(CV_SCORING if scoring is None else scoring)
    res = cross_validate(
        estimator, X, y, cv=cv, groups=groups, scoring=scoring,
        n_jobs=n_jobs, error_score="raise", return_train_score=True,
    )
    folds = pd.DataFrame(res)
    mean, std = {}, {}
    for name in scoring:
        col = f"test_{name}"
        vals = folds[col].to_numpy(dtype="float64")
        if name.startswith("neg_"):
            name, vals = name[4:], -vals
        mean[name] = float(vals.mean())
        std[name] = float(vals.std(ddof=0))
    return CVResult(mean=mean, std=std, folds=folds, label=label)


def metrics_frame(results: Mapping[str, Mapping[str, float]],
                  columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Converte ``{modelo: metricas}`` em DataFrame.

    Sem ``columns``: ordem canônica (`METRIC_ORDER`) seguida das demais colunas.
    Com ``columns``: **apenas** as colunas pedidas (tabelas de relatório).
    """
    df = pd.DataFrame(results).T
    if columns is not None:
        return df[[c for c in columns if c in df.columns]]
    cols = [c for c in METRIC_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Markdown de um DataFrame numérico sem depender do `tabulate`."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else floatfmt.format(v))
        else:
            d[c] = d[c].astype(str)
    header = "| " + " | ".join([d.index.name or ""] + list(d.columns)) + " |"
    sep = "|" + "|".join(["---"] * (len(d.columns) + 1)) + "|"
    rows = ["| " + " | ".join([str(i)] + list(r)) + " |" for i, r in zip(d.index, d.to_numpy())]
    return "\n".join([header, sep, *rows])
