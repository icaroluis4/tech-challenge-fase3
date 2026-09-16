"""Gráficos padronizados do projeto (matplotlib/seaborn/plotly)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Gráficos de avaliação de classificadores (usados pelos modelos A e B)
# --------------------------------------------------------------------------

def plot_roc_pr(curvas: dict[str, tuple[np.ndarray, np.ndarray]],
                titulo: str = "Desempenho no holdout") -> plt.Figure:
    """Painel ROC + Precision-Recall para vários modelos.

    `curvas` mapeia nome → (y_true, proba).
    """
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for nome, (y, p) in curvas.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=1.8, label=f"{nome} (AUC={roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, lw=1.8, label=f"{nome} (AP={average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="aleatório")
    axes[0].set(xlabel="Falso positivo", ylabel="Verdadeiro positivo", title="Curva ROC")
    base = float(np.mean(next(iter(curvas.values()))[0]))
    axes[1].axhline(base, color="k", ls="--", lw=1, label=f"prevalência={base:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precisão", title="Curva Precision-Recall")
    for ax in axes:
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plot_calibration(tabelas: dict[str, pd.DataFrame],
                     titulo: str = "Curva de calibração") -> plt.Figure:
    """Curva de calibração a partir de `metrics.calibration_table`."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibração perfeita")
    for nome, tab in tabelas.items():
        ece = tab.attrs.get("ece", float("nan"))
        ax.plot(tab["p_media"], tab["freq_observada"], "o-", lw=1.8,
                label=f"{nome} (ECE={ece:.4f})")
    ax.set(xlabel="Probabilidade média prevista", ylabel="Frequência observada",
           title=titulo)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, labels: tuple[str, str],
                   titulo: str = "Matriz de confusão") -> plt.Figure:
    """Heatmap da matriz de confusão com contagem e percentual da linha."""
    cm = np.asarray(cm)
    perc = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]:,}\n({perc[i, j]:.1%})" for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(perc, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, vmin=0, vmax=1)
    ax.set(xlabel="Previsto", ylabel="Observado", title=titulo)
    fig.tight_layout()
    return fig


def plot_learning_curve(x: np.ndarray, treino: np.ndarray, validacao: np.ndarray,
                        xlabel: str = "n_estimators", ylabel: str = "ROC-AUC",
                        titulo: str = "Curva treino × validação") -> plt.Figure:
    """Curva de overfitting: métrica em treino e validação ao longo da capacidade."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, treino, "o-", lw=1.8, label="treino")
    ax.plot(x, validacao, "s-", lw=1.8, label="validação")
    gap = np.asarray(treino) - np.asarray(validacao)
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=f"{titulo} (gap final={gap[-1]:.4f})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_bars(df: pd.DataFrame, metricas: list[str],
                     titulo: str = "Comparação de modelos") -> plt.Figure:
    """Barras horizontais comparando modelos em várias métricas."""
    long = (df[metricas].reset_index(names="modelo")
            .melt(id_vars="modelo", var_name="metrica", value_name="valor"))
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(df) * len(metricas) ** 0.5 + 2))
    sns.barplot(long, y="modelo", x="valor", hue="metrica", ax=ax)
    ax.set(xlabel="valor", ylabel="", title=titulo)
    ax.legend(title="", fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
