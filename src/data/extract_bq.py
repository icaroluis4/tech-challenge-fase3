"""Extração BigQuery -> Parquet local (idempotente).

Regras:
- Nunca varrer a Bronze em loop: cada extract salva parquet e, se o arquivo
  já existe, não reexecuta a query (a menos que force=True).
- SQL sempre em arquivo .py (nunca `python -c` com crases no PowerShell).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from src.config import (
    DATA_RAW,
    GCP_PROJECT_ID,
    REPORTS_DIR,
    TBL_ALUNO,
    TBL_GOLD_MUNICIPIO,
    TBL_GOLD_UF,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _client() -> bigquery.Client:
    return bigquery.Client(project=GCP_PROJECT_ID)


def _query_to_parquet(sql: str, out: Path, force: bool = False) -> pd.DataFrame:
    if out.exists() and not force:
        log.info("cache hit: %s", out.name)
        return pd.read_parquet(out)
    log.info("querying BQ -> %s", out.name)
    df = _client().query(sql).to_dataframe()
    df.to_parquet(out, index=False)
    log.info("saved %s (%d linhas, %d cols)", out.name, len(df), df.shape[1])
    return df


def extract_gold_municipio(force: bool = False) -> pd.DataFrame:
    sql = f"SELECT * FROM `{TBL_GOLD_MUNICIPIO}`"
    return _query_to_parquet(sql, DATA_RAW / "gold_municipio.parquet", force)


def extract_gold_uf(force: bool = False) -> pd.DataFrame:
    sql = f"SELECT * FROM `{TBL_GOLD_UF}`"
    return _query_to_parquet(sql, DATA_RAW / "gold_uf.parquet", force)


def extract_externos(force: bool = False) -> pd.DataFrame:
    """Enriquecimento externo via Base dos Dados (uma query só)."""
    sql = f"""
    WITH g AS (
      SELECT DISTINCT CAST(co_municipio AS STRING) AS id_municipio
      FROM `{TBL_GOLD_MUNICIPIO}`
    )
    SELECT g.id_municipio,
           pop.populacao AS populacao_2024,
           pib.pib,
           va.va, va.va_agropecuaria, va.va_industria, va.va_servicos,
           adh.idhm, adh.idhm_e, adh.idhm_l, adh.idhm_r, adh.indice_gini, adh.renda_pc,
           adh.taxa_analfabetismo_15_mais, adh.expectativa_anos_estudo,
           dir.nome_regiao_imediata, dir.nome_regiao_intermediaria, dir.capital_uf
    FROM g
    LEFT JOIN (SELECT id_municipio, populacao
               FROM `basedosdados.br_ibge_populacao.municipio` WHERE ano = 2024) pop
      USING (id_municipio)
    -- PIB total: último ano disponível (2023)
    LEFT JOIN (SELECT id_municipio, pib
               FROM `basedosdados.br_ibge_pib.municipio` WHERE ano = 2023) pib
      USING (id_municipio)
    -- Valor adicionado setorial: só existe até 2021 na Base dos Dados (2022/2023 nulos)
    LEFT JOIN (SELECT id_municipio, va, va_agropecuaria, va_industria, va_servicos
               FROM `basedosdados.br_ibge_pib.municipio` WHERE ano = 2021) va
      USING (id_municipio)
    LEFT JOIN (SELECT id_municipio, idhm, idhm_e, idhm_l, idhm_r, indice_gini, renda_pc,
                      taxa_analfabetismo_15_mais, expectativa_anos_estudo
               FROM `basedosdados.mundo_onu_adh.municipio` WHERE ano = 2010) adh
      USING (id_municipio)
    LEFT JOIN (SELECT id_municipio, nome_regiao_imediata, nome_regiao_intermediaria, capital_uf
               FROM `basedosdados.br_bd_diretorios_brasil.municipio`) dir
      USING (id_municipio)
    """
    df = _query_to_parquet(sql, DATA_RAW / "externos_municipio.parquet", force)
    # cobertura por fonte
    cov = {
        "populacao_2024": df["populacao_2024"].notna().mean(),
        "pib": df["pib"].notna().mean(),
        "va_2021": df["va"].notna().mean(),
        "idhm": df["idhm"].notna().mean(),
        "diretorios": df["nome_regiao_intermediaria"].notna().mean(),
    }
    for k, v in cov.items():
        log.info("cobertura %-16s %.1f%%", k, 100 * v)
    return df


def extract_aluno(force: bool = False) -> pd.DataFrame:
    """Alunos presentes (IN_PRESENCA_LP=1) — query única, colunas mínimas."""
    sql = f"""
    SELECT ID_ALUNO, CO_UF, SG_UF, CO_MUNICIPIO, TP_DEPENDENCIA, IN_ALFABETIZADO
    FROM `{TBL_ALUNO}`
    WHERE IN_PRESENCA_LP = 1 AND IN_ALFABETIZADO IS NOT NULL
    """
    return _query_to_parquet(sql, DATA_RAW / "aluno_presente.parquet", force)


def evidencia_leakage_proficiencia(force: bool = False) -> None:
    """Auditoria F1: IN_ALFABETIZADO é determinístico de VL_PROFICIENCIA_LP (corte 743)."""
    out = REPORTS_DIR / "evidencia_leakage_proficiencia.md"
    if out.exists() and not force:
        log.info("cache hit: %s", out.name)
        return
    sql = f"""
    SELECT IN_ALFABETIZADO,
           COUNT(*) AS n,
           MIN(VL_PROFICIENCIA_LP) AS min_prof,
           MAX(VL_PROFICIENCIA_LP) AS max_prof
    FROM `{TBL_ALUNO}`
    WHERE IN_PRESENCA_LP = 1 AND IN_ALFABETIZADO IS NOT NULL
    GROUP BY IN_ALFABETIZADO
    ORDER BY IN_ALFABETIZADO
    """
    df = _client().query(sql).to_dataframe()
    lines = [
        "# Evidência de leakage — proficiência determina o target",
        "",
        "Consulta sobre `bronze_alfabetizacao.aeeb_ts_aluno` (apenas presentes):",
        "",
        "| IN_ALFABETIZADO | n | min proficiência | max proficiência |",
        "|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['IN_ALFABETIZADO'])} | {int(r['n']):,} | "
            f"{r['min_prof']:.2f} | {r['max_prof']:.2f} |"
        )
    lines += [
        "",
        "**Conclusão:** `IN_ALFABETIZADO = 1` ⇔ `VL_PROFICIENCIA_LP >= 743,0`.",
        "Proficiência, respostas, gabaritos, caderno, blocos e peso do aluno são",
        "**leakage total** e estão proibidos como features (ver `src/preprocessing/leakage.py`).",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("saved %s", out.name)


def run_all(force: bool = False) -> None:
    extract_gold_municipio(force)
    extract_gold_uf(force)
    extract_externos(force)
    extract_aluno(force)
    evidencia_leakage_proficiencia(force)


if __name__ == "__main__":
    run_all()
