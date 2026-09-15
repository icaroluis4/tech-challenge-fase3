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
    for col in ["populacao_2024", "pib", "va", "va_agropecuaria", "idhm", "share_va_agro"]:
        assert feats[col].notna().mean() >= 0.98, f"cobertura baixa em {col}"


def test_nenhuma_feature_totalmente_nula(feats):
    vazias = [c for c in feats.columns if feats[c].isna().all()]
    assert not vazias, f"features 100% nulas: {vazias}"


def test_escala_pib_per_capita(feats):
    # PIB per capita municipal BR (2023) em R$: mediana ~ 20–40 mil
    med = feats["pib_per_capita_2023"].median()
    assert 15_000 < med < 60_000, f"mediana fora de escala: {med}"
    assert feats["share_va_agro"].quantile(0.99) <= 1.0


def test_derivadas_coerentes(feats):
    d = feats["delta_2023_2024"] - (feats["pc_alfabetizado_2024"] - feats["pc_alfabetizado_2023"])
    assert d.abs().max() < 1e-6
    assert feats["regiao"].isin(["N", "NE", "CO", "SE", "S"]).all()
