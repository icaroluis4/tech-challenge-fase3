"""Listas negras de leakage e guarda de conformidade.

Fatos verificados (ver PLANO_IMPLEMENTACAO.md §0.2):
- F1: IN_ALFABETIZADO é determinístico de VL_PROFICIENCIA_LP (corte 743,0).
- F6: colunas de resultado 2025 da Gold são leakage para targets de 2025.
"""
from __future__ import annotations

from typing import Iterable

LEAKAGE_ALUNO = [
    "VL_PROFICIENCIA_LP", "VL_PESO_ALUNO_LP", "CO_CADERNO_LP",
    "CO_BLOCO_1", "CO_BLOCO_2", "CO_BLOCO_3", "CO_BLOCO_4",
    "TX_RESPOSTA_BLOCO_1", "TX_RESPOSTA_BLOCO_2", "TX_RESPOSTA_BLOCO_3", "TX_RESPOSTA_BLOCO_4",
    "TX_GABARITO_BLOCO_1", "TX_GABARITO_BLOCO_2", "TX_GABARITO_BLOCO_3", "TX_GABARITO_BLOCO_4",
    "IN_PREENCHIMENTO_LP", "IN_PRESENCA_LP",  # presença define o filtro, não é feature
]

LEAKAGE_2025 = [
    "pc_alfabetizado_aeeb_2025", "vl_media_lp",
    "pc_nivel_0", "pc_nivel_1", "pc_nivel_2", "pc_nivel_3", "pc_nivel_4",
    "pc_nivel_5", "pc_nivel_6", "pc_nivel_7", "pc_nivel_8",
    "pc_alfabetizado_metas_2025", "gap_meta_2025", "atingiu_meta_2025",
    "evolucao_2023_2025", "pc_participacao", "co_nivel_alfabetizacao",
]

IDS = ["ID_ALUNO", "ID_ESCOLA", "id_aluno", "id_escola",  # dataset_aluno é lowercase
       "co_municipio", "no_municipio", "NO_MUNICIPIO", "nu_ano_avaliacao", "id_municipio"]

BLACKLIST = set(LEAKAGE_ALUNO) | set(LEAKAGE_2025) | set(IDS)


def assert_no_leakage(columns: Iterable[str], allow: Iterable[str] | None = None) -> None:
    """Falha se qualquer coluna proibida estiver presente em `columns`.

    `allow` permite exceções explícitas (ex.: co_municipio como chave de join
    antes de ser removida de X).
    """
    allow_set = set(allow or [])
    bad = sorted((set(columns) & BLACKLIST) - allow_set)
    if bad:
        raise AssertionError(f"Colunas proibidas (leakage/ID) detectadas: {bad}")
