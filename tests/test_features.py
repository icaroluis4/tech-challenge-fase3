"""Validações da feature store municipal (roda após build_features)."""
import pandas as pd
import pytest

from src.config import DATA_PROCESSED
from src.preprocessing.leakage import assert_no_leakage

FEATS = DATA_PROCESSED / "features_municipio.parquet"

pytestmark = pytest.mark.skipif(not FEATS.exists(), reason="features ainda não geradas")


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    return pd.read_parquet(FEATS)


def test_grao_unico_municipio(feats):
    assert feats["co_municipio"].is_unique
    assert len(feats) == 5500


def test_sem_leakage(feats):
    assert_no_leakage(feats.columns, allow={"co_municipio", "no_municipio"})


def test_cobertura_externos(feats):
    for col in ["populacao_2024", "pib", "idhm"]:
        assert feats[col].notna().mean() >= 0.98, f"cobertura baixa em {col}"


def test_derivadas_coerentes(feats):
    d = feats["delta_2023_2024"] - (feats["pc_alfabetizado_2024"] - feats["pc_alfabetizado_2023"])
    assert d.abs().max() < 1e-6
    assert feats["regiao"].isin(["N", "NE", "CO", "SE", "S"]).all()
