"""Estratégias de divisão treino/teste e validação cruzada.

Duas visões de generalização são reportadas em todos os modelos:
- **aleatória estratificada** (StratifiedKFold): desempenho médio;
- **territorial** (GroupKFold por município ou UF): desempenho em território
  nunca visto — o número que interessa a um gestor público.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from src.config import SEED

TARGET_ALUNO = "in_alfabetizado"
TARGET_MUNICIPIO = "atingiu_meta_2025"
GROUP_ALUNO = "co_municipio"
GROUP_MUNICIPIO = "sg_uf"


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series | None = None
    groups_test: pd.Series | None = None

    def summary(self) -> str:
        pos_tr = float(self.y_train.mean())
        pos_te = float(self.y_test.mean())
        return (f"train={len(self.y_train):,} (pos={pos_tr:.3f}) | "
                f"test={len(self.y_test):,} (pos={pos_te:.3f})")


def _holdout(df: pd.DataFrame, target: str, group: str | None,
             test_size: float, seed: int) -> Split:
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=y, random_state=seed
    )
    groups = df[group] if group and group in df.columns else None
    return Split(
        X_train=X.iloc[idx_tr].reset_index(drop=True),
        X_test=X.iloc[idx_te].reset_index(drop=True),
        y_train=y.iloc[idx_tr].reset_index(drop=True),
        y_test=y.iloc[idx_te].reset_index(drop=True),
        groups_train=None if groups is None else groups.iloc[idx_tr].reset_index(drop=True),
        groups_test=None if groups is None else groups.iloc[idx_te].reset_index(drop=True),
    )


def split_aluno(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED) -> Split:
    """Holdout 80/20 estratificado por `in_alfabetizado`; grupos = `co_municipio`."""
    return _holdout(df, TARGET_ALUNO, GROUP_ALUNO, test_size, seed)


def split_municipio(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED) -> Split:
    """Holdout 80/20 estratificado por `atingiu_meta_2025` (nulos descartados);
    grupos = `sg_uf`."""
    d = df.dropna(subset=[TARGET_MUNICIPIO]).copy()
    d[TARGET_MUNICIPIO] = d[TARGET_MUNICIPIO].astype(int)
    return _holdout(d, TARGET_MUNICIPIO, GROUP_MUNICIPIO, test_size, seed)


def stratified_cv(n_splits: int = 5, seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def group_cv(n_splits: int = 5) -> GroupKFold:
    """GroupKFold: garante que nenhum grupo (município/UF) apareça em treino e
    validação ao mesmo tempo."""
    return GroupKFold(n_splits=n_splits)


def sample_aluno(df: pd.DataFrame, n: int = 400_000, seed: int = SEED,
                 strata: tuple[str, str] = ("sg_uf", TARGET_ALUNO)) -> pd.DataFrame:
    """Amostra estratificada por (UF, target) para experimentação rápida.
    Se `n >= len(df)`, retorna o df inteiro."""
    if n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    return (
        df.groupby(list(strata), observed=True)
          .sample(frac=frac, random_state=seed)
          .reset_index(drop=True)
    )
