"""Construção da feature store municipal e do dataset de aluno.

Saídas em data/processed/:
- features_municipio.parquet  (sem leakage 2025; só info disponível antes da avaliação 2025)
- targets_municipio.parquet   (targets 2025 separados: atingiu_meta_2025 etc.)
- dataset_aluno.parquet       (aluno presente + contexto municipal)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW
from src.preprocessing.leakage import LEAKAGE_2025, assert_no_leakage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_REGIAO = {
    "N": {"11", "12", "13", "14", "15", "16", "17"},
    "NE": {"21", "22", "23", "24", "25", "26", "27", "28", "29"},
    "CO": {"50", "51", "52", "53"},
    "SE": {"31", "32", "33", "35"},
    "S": {"41", "42", "43"},
}
_UF2REGIAO = {uf: reg for reg, ufs in _REGIAO.items() for uf in ufs}

TARGETS_MUNICIPIO = ["atingiu_meta_2025", "pc_alfabetizado_aeeb_2025", "gap_meta_2025"]


def _porte(pop: pd.Series) -> pd.Categorical:
    bins = [0, 5_000, 20_000, 100_000, 500_000, np.inf]
    labels = ["pequeno_1", "pequeno_2", "medio", "grande", "metropole"]
    return pd.cut(pop, bins=bins, labels=labels)


def build_features_municipio() -> pd.DataFrame:
    gold = pd.read_parquet(DATA_RAW / "gold_municipio.parquet")
    ext = pd.read_parquet(DATA_RAW / "externos_municipio.parquet")
    gold["id_municipio"] = gold["co_municipio"].astype(str)
    df = gold.merge(ext, on="id_municipio", how="left", validate="1:1")
    log.info("gold %d + externos -> %d linhas", len(gold), len(df))

    # --- derivadas: histórico e metas ---
    df["delta_2023_2024"] = df["pc_alfabetizado_2024"] - df["pc_alfabetizado_2023"]
    df["gap_meta_2024"] = df["pc_alfabetizado_2024"] - df["meta_2024"]
    df["atingiu_meta_2024"] = (df["pc_alfabetizado_2024"] >= df["meta_2024"]).astype("float")
    df.loc[df["meta_2024"].isna() | df["pc_alfabetizado_2024"].isna(), "atingiu_meta_2024"] = np.nan
    df["esforco_2025"] = df["meta_2025"] - df["pc_alfabetizado_2024"]
    df["ambicao_2030"] = df["meta_2030"] - df["pc_alfabetizado_2024"]

    # --- rede escolar ---
    df["pct_escolas_rurais"] = df["qtd_escolas_rurais"] / df["qtd_escolas"]
    df["pct_escolas_privadas"] = df["qtd_escolas_privadas"] / df["qtd_escolas"]
    df["alunos_por_turma_ai"] = df["total_mat_fund_ai"] / df["total_tur_fund_ai"]

    # --- território ---
    df["regiao"] = df["co_uf"].astype("Int64").astype(str).str.zfill(2).map(_UF2REGIAO)

    # --- populacional / socioeconômico ---
    df["log_populacao"] = np.log1p(df["populacao_2024"])
    df["porte"] = _porte(df["populacao_2024"])
    # pib da Base dos Dados já está em R$ correntes (média BR ≈ R$ 39 mil/hab)
    df["pib_per_capita_2023"] = df["pib"] / df["populacao_2024"]
    # VA setorial só existe até 2021 na fonte (ver extract_externos)
    df["share_va_agro"] = df["va_agropecuaria"] / df["va"]

    # --- separa targets 2025 (nunca entram como feature) ---
    targets = df[["co_municipio"] + TARGETS_MUNICIPIO].copy()
    targets.to_parquet(DATA_PROCESSED / "targets_municipio.parquet", index=False)

    drop_cols = set(LEAKAGE_2025) | {"id_municipio", "nu_ano_avaliacao"}
    feats = df.drop(columns=[c for c in drop_cols if c in df.columns])
    assert_no_leakage(feats.columns, allow={"co_municipio", "no_municipio"})
    feats.to_parquet(DATA_PROCESSED / "features_municipio.parquet", index=False)
    log.info("features_municipio: %d linhas, %d cols", len(feats), feats.shape[1])
    return feats


def build_dataset_aluno(feats: pd.DataFrame | None = None) -> pd.DataFrame:
    if feats is None:
        feats = pd.read_parquet(DATA_PROCESSED / "features_municipio.parquet")
    aluno = pd.read_parquet(DATA_RAW / "aluno_presente.parquet")
    aluno = aluno.rename(columns=str.lower)  # CO_MUNICIPIO -> co_municipio etc.
    # evita sufixos _x/_y: colunas do aluno são redundantes com as da feature store
    aluno = aluno.drop(columns=["co_uf", "sg_uf"])
    n0 = len(aluno)
    df = aluno.merge(feats, on="co_municipio", how="inner", validate="m:1")
    log.info("aluno: %d -> %d após join (descartados %.2f%%)",
             n0, len(df), 100 * (1 - len(df) / n0))
    df["tp_dependencia"] = df["tp_dependencia"].astype("category")
    df.to_parquet(DATA_PROCESSED / "dataset_aluno.parquet", index=False)
    log.info("dataset_aluno: %d linhas, %d cols", len(df), df.shape[1])
    return df


def materializar_gold_ml(feats: pd.DataFrame | None = None) -> None:
    """Bônus: materializa a feature store no BQ (dataset gold_ml) — continuidade medalhão."""
    from google.cloud import bigquery

    from src.config import BQ_DATASET_ML, GCP_PROJECT_ID

    if feats is None:
        feats = pd.read_parquet(DATA_PROCESSED / "features_municipio.parquet")
    client = bigquery.Client(project=GCP_PROJECT_ID)
    ds = bigquery.Dataset(f"{GCP_PROJECT_ID}.{BQ_DATASET_ML}")
    ds.location = "US"
    client.create_dataset(ds, exists_ok=True)
    table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ML}.features_municipio"
    job = client.load_table_from_dataframe(
        feats, table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    log.info("materializado %s (%d linhas)", table_id, len(feats))


def run_all() -> None:
    feats = build_features_municipio()
    build_dataset_aluno(feats)
    materializar_gold_ml(feats)


if __name__ == "__main__":
    run_all()
