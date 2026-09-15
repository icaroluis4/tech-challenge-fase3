"""Guard anti-leakage: nenhuma coluna proibida pode entrar em X."""
import pytest

from src.preprocessing.leakage import (
    IDS,
    LEAKAGE_2025,
    LEAKAGE_ALUNO,
    assert_no_leakage,
)


def test_listas_nao_vazias():
    assert LEAKAGE_ALUNO and LEAKAGE_2025 and IDS


def test_assert_passa_com_features_legitimas():
    assert_no_leakage(["pc_alfabetizado_2024", "idhm", "sg_uf", "qtd_escolas"])


@pytest.mark.parametrize("col", LEAKAGE_ALUNO + LEAKAGE_2025 + IDS)
def test_assert_falha_com_coluna_proibida(col):
    with pytest.raises(AssertionError):
        assert_no_leakage(["pc_alfabetizado_2024", col])


def test_allow_permite_chave_de_join():
    assert_no_leakage(["co_municipio", "idhm"], allow={"co_municipio"})
