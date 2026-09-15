"""Construção do pré-processador (ColumnTransformer) usado por todos os modelos.

Requisitos do challenge atendidos aqui:
- imputação de numéricas (mediana) e categóricas (moda);
- transformação numérica (log1p em contagens de cauda longa + StandardScaler)
  e categórica (OneHotEncoder);
- tudo ocorre DENTRO do Pipeline do estimador → só aprende no `fit` do treino.

Uso típico::

    num_cols, cat_cols = infer_feature_columns(X)
    pre = build_preprocessor(num_cols, cat_cols)
    pipe = Pipeline([("pre", pre), ("model", LogisticRegression())])
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.preprocessing.leakage import BLACKLIST, assert_no_leakage

# Categóricas conhecidas do projeto (município e aluno).
CATEGORICAL_COLS: list[str] = ["sg_uf", "regiao", "porte", "capital_uf", "tp_dependencia"]

# Colunas texto de alta cardinalidade que NÃO entram no modelo (só descrição).
DESCRIPTIVE_COLS: list[str] = ["nome_regiao_imediata", "nome_regiao_intermediaria"]

# Contagens de cauda longa → log1p antes do scaler.
LOG1P_COLS: list[str] = [
    "populacao_2024", "total_mat_fund_ai", "total_mat_1_2_ano", "qtd_escolas",
    "qtd_escolas_municipais", "qtd_escolas_estaduais", "qtd_escolas_privadas",
    "qtd_escolas_urbanas", "qtd_escolas_rurais", "total_doc_fund_ai",
    "total_tur_fund_ai", "pib", "va", "va_agropecuaria", "va_industria", "va_servicos",
]


def _to_float(X):
    """Converte dtypes nullable do pandas (Int64/Float64/boolean) para float64 com NaN."""
    if isinstance(X, pd.DataFrame):
        return X.astype("float64")
    return np.asarray(X, dtype="float64")


def _symlog1p(X):
    """log1p simétrico: sign(x)·log1p(|x|). Igual a log1p para x ≥ 0, mas
    não gera NaN em contagens/valores negativos (ex.: VA agropecuária < 0)."""
    X = np.asarray(X, dtype="float64")
    return np.sign(X) * np.log1p(np.abs(X))


def _to_str(X):
    """Categóricas → object/str para o OneHotEncoder (NaN preservado)."""
    if isinstance(X, pd.DataFrame):
        return X.astype("object").where(X.notna(), np.nan)
    return X


def infer_feature_columns(
    df: pd.DataFrame,
    target: str | None = None,
    extra_drop: Iterable[str] = (),
    categorical: Sequence[str] = CATEGORICAL_COLS,
) -> tuple[list[str], list[str]]:
    """Separa colunas numéricas e categóricas de `df`, já removendo BLACKLIST,
    target, descritivas e `extra_drop`. Levanta erro se restar leakage."""
    drop = set(BLACKLIST) | set(DESCRIPTIVE_COLS) | set(extra_drop)
    if target:
        drop.add(target)
    cols = [c for c in df.columns if c not in drop]
    assert_no_leakage(cols)
    cat_cols = [c for c in cols if c in categorical]
    num_cols = [c for c in cols if c not in cat_cols]
    # Segurança: numéricas precisam ser realmente numéricas/booleanas.
    bad = [c for c in num_cols if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    if bad:
        raise TypeError(f"Colunas não numéricas fora da lista categórica: {bad}")
    return num_cols, cat_cols


def build_preprocessor(
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    scale: bool = True,
    log_cols: Sequence[str] | None = None,
    min_frequency: int | float | None = 20,
) -> ColumnTransformer:
    """ColumnTransformer com três ramos:

    - ``log``: colunas de contagem (LOG1P_COLS ∩ num_cols) → mediana → log1p → scaler
    - ``num``: demais numéricas → mediana → scaler
    - ``cat``: categóricas → moda → OneHotEncoder(handle_unknown="ignore")

    `scale=False` deixa passthrough (útil para árvores, que não precisam de escala).
    """
    log_cols = [c for c in (LOG1P_COLS if log_cols is None else log_cols) if c in num_cols]
    plain_cols = [c for c in num_cols if c not in log_cols]

    def _scaler():
        return StandardScaler() if scale else "passthrough"

    log_pipe = Pipeline([
        ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(_symlog1p, feature_names_out="one-to-one")),
        ("scaler", _scaler()),
    ])
    num_pipe = Pipeline([
        ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", _scaler()),
    ])
    cat_pipe = Pipeline([
        ("to_str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency,
                              sparse_output=False)),
    ])

    transformers = []
    if log_cols:
        transformers.append(("log", log_pipe, list(log_cols)))
    if plain_cols:
        transformers.append(("num", num_pipe, list(plain_cols)))
    if cat_cols:
        transformers.append(("cat", cat_pipe, list(cat_cols)))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="default")
    return pre


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Nomes das colunas após o `fit` (para importâncias / SHAP)."""
    return list(preprocessor.get_feature_names_out())
