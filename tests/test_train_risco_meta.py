"""Modelo B (risco de meta): matriz genérica em t, candidatos, ablação e
execução fim a fim em dados sintéticos pequenos.

Os testes NÃO dependem dos parquets reais (gitignored) — usam um gerador
sintético que reproduz o desenho do problema: target municipal `atingiu_meta_2025`
com sinal no histórico (`resultado_t1`) e no contexto socioeconômico.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.modeling.predict_2026 import _sanidade
from src.modeling.train_risco_meta import (
    EXTRA_DROP,
    HISTORICO_COLS,
    TrainConfig,
    _ablacao_sem_historico,
    carregar_dados,
    make_candidates,
    montar_matriz_t,
    param_distributions,
    treinar,
)
from src.preprocessing.leakage import assert_no_leakage
from src.preprocessing.splits import TARGET_MUNICIPIO, split_municipio
from src.preprocessing.transformers import infer_feature_columns

N_MUN = 300


@pytest.fixture(scope="module")
def municipio_synth() -> tuple[pd.DataFrame, pd.DataFrame]:
    """300 municípios com histórico 2023/2024, metas e contexto; target 2025
    depende de `pc_2024` (inércia) e de `idhm_e` (contexto)."""
    rng = np.random.default_rng(SEED)
    ufs = ["SP", "BA", "CE", "AM", "RS"]
    f = pd.DataFrame({
        "co_municipio": pd.array(rng.choice(np.arange(1_100_000, 5_300_000), N_MUN,
                                            replace=False), dtype="Int64"),
        "no_municipio": [f"m{i}" for i in range(N_MUN)],
        "sg_uf": rng.choice(ufs, N_MUN),
        "co_uf": pd.array(rng.integers(11, 53, N_MUN), dtype="Int64"),
        "pc_alfabetizado_2023": rng.uniform(20, 85, N_MUN),
        "pc_alfabetizado_2024": rng.uniform(25, 90, N_MUN),
        "meta_2024": rng.uniform(30, 90, N_MUN),
        "meta_2025": rng.uniform(35, 95, N_MUN),
        "meta_2026": rng.uniform(40, 97, N_MUN),
        "meta_2030": rng.uniform(60, 100, N_MUN),
        "delta_2023_2024": rng.uniform(-5, 10, N_MUN),
        "gap_meta_2024": rng.uniform(-20, 20, N_MUN),
        "atingiu_meta_2024": pd.array(rng.integers(0, 2, N_MUN), dtype="boolean"),
        "esforco_2025": rng.uniform(0, 30, N_MUN),
        "ambicao_2030": rng.uniform(0, 40, N_MUN),
        "populacao_2024": pd.array(rng.integers(1_000, 3_000_000, N_MUN), dtype="Int64"),
        "qtd_escolas": pd.array(rng.integers(1, 400, N_MUN), dtype="Int64"),
        "idhm_e": rng.uniform(0.3, 0.9, N_MUN),
        "renda_pc": rng.uniform(150, 2_200, N_MUN),
        "pct_escolas_rurais": pd.array(rng.uniform(0, 1, N_MUN), dtype="Float64"),
        "va_agropecuaria": rng.normal(2e5, 3e5, N_MUN),
        "regiao": rng.choice(["N", "NE", "CO", "SE", "S"], N_MUN),
        "porte": pd.Categorical(rng.choice(["pequeno_1", "medio", "grande"], N_MUN)),
        "capital_uf": pd.array(rng.integers(0, 2, N_MUN), dtype="Int64"),
        "nome_regiao_intermediaria": rng.choice([f"ri{i}" for i in range(8)], N_MUN),
    })
    for c in ["pc_alfabetizado_2023", "idhm_e", "porte", "capital_uf"]:
        f.loc[rng.random(N_MUN) < 0.1, c] = pd.NA

    # target 2025: inércia do resultado + contexto; ~30% não atinge
    p_atinge = 1 / (1 + np.exp(-(
        0.10 * (f["pc_alfabetizado_2024"] - f["meta_2025"])
        + 2.0 * (f["idhm_e"].fillna(0.6) - 0.6)
    )))
    t = pd.DataFrame({
        "co_municipio": f["co_municipio"],
        TARGET_MUNICIPIO: pd.array(rng.binomial(1, p_atinge), dtype="boolean"),
        "pc_alfabetizado_aeeb_2025": np.clip(
            f["pc_alfabetizado_2024"] + rng.normal(2, 4, N_MUN), 0, 100),
        "gap_meta_2025": rng.uniform(-15, 15, N_MUN),
    })
    # 83 municípios sem rótulo no real — aqui, ~5%
    t.loc[rng.random(N_MUN) < 0.05, TARGET_MUNICIPIO] = pd.NA
    return f, t


@pytest.fixture(scope="module")
def matriz_2025(municipio_synth):
    f, t = municipio_synth
    return montar_matriz_t(f, t, t=2025)


@pytest.fixture(scope="module")
def split_synth(matriz_2025):
    return split_municipio(matriz_2025, seed=SEED)


@pytest.fixture(scope="module")
def cols_synth(split_synth):
    return infer_feature_columns(split_synth.X_train, extra_drop=EXTRA_DROP)


# --------------------------------------------------------------------------
# montar_matriz_t
# --------------------------------------------------------------------------

def test_matriz_2025_tem_bloco_generico_e_target(matriz_2025):
    for c in HISTORICO_COLS:
        assert c in matriz_2025.columns, c
    assert TARGET_MUNICIPIO in matriz_2025.columns
    assert len(matriz_2025) == N_MUN


def test_matriz_2025_dropa_colunas_de_ano_explicito(matriz_2025):
    for c in ["pc_alfabetizado_2023", "pc_alfabetizado_2024", "meta_2024",
              "meta_2025", "meta_2026", "meta_2030", "delta_2023_2024",
              "gap_meta_2024", "esforco_2025", "ambicao_2030", "atingiu_meta_2024"]:
        assert c not in matriz_2025.columns, c


def test_matriz_2025_sem_leakage(matriz_2025):
    assert_no_leakage(matriz_2025.columns,
                      allow=[TARGET_MUNICIPIO, "co_municipio", "no_municipio"])


def test_matriz_2025_valores_coerentes(municipio_synth):
    f, t = municipio_synth
    m = montar_matriz_t(f, t, t=2025)
    assert np.allclose(m["resultado_t1"], f["pc_alfabetizado_2024"], equal_nan=True)
    assert np.allclose(m["resultado_t2"], f["pc_alfabetizado_2023"], equal_nan=True)
    assert np.allclose(m["delta_t1"],
                       f["pc_alfabetizado_2024"] - f["pc_alfabetizado_2023"],
                       equal_nan=True)
    assert np.allclose(m["meta_t"], f["meta_2025"], equal_nan=True)
    assert np.allclose(m["esforco_t"],
                       f["meta_2025"] - f["pc_alfabetizado_2024"], equal_nan=True)


def test_matriz_2026_sem_target_e_com_resultado_2025(municipio_synth):
    f, t = municipio_synth
    m = montar_matriz_t(f, t, t=2026)
    assert TARGET_MUNICIPIO not in m.columns
    assert_no_leakage(m.columns, allow=["co_municipio", "no_municipio"])
    assert np.allclose(m["resultado_t1"], t["pc_alfabetizado_aeeb_2025"],
                       equal_nan=True)
    assert np.allclose(m["resultado_t2"], f["pc_alfabetizado_2024"], equal_nan=True)
    assert np.allclose(m["meta_t"], f["meta_2026"], equal_nan=True)
    assert np.allclose(m["esforco_t"],
                       f["meta_2026"] - t["pc_alfabetizado_aeeb_2025"], equal_nan=True)
    # atingiu_t1 em 2026 = atingiu_meta_2025 (pode ter NA)
    comp = (m["atingiu_t1"].astype("Float64")
            == t[TARGET_MUNICIPIO].astype("Float64"))
    assert comp.fillna(True).all()


def test_matriz_ano_invalido(municipio_synth):
    f, t = municipio_synth
    with pytest.raises(ValueError, match="ano-alvo"):
        montar_matriz_t(f, t, t=2027)


# --------------------------------------------------------------------------
# Features / candidatos
# --------------------------------------------------------------------------

def test_features_sem_leakage_nem_ids(cols_synth):
    num, cat = cols_synth
    assert_no_leakage(num + cat)
    assert "co_municipio" not in num + cat
    assert "no_municipio" not in num + cat
    assert "co_uf" not in num + cat
    assert "sg_uf" in cat
    assert TARGET_MUNICIPIO not in num + cat
    for c in HISTORICO_COLS:
        assert c in num + cat, c


def test_candidatos_esperados(cols_synth):
    num, cat = cols_synth
    cands = make_candidates(num, cat)
    assert {"dummy", "logreg", "hist_gb", "random_forest"} <= set(cands)
    assert all(isinstance(p, Pipeline) for p in cands.values())


def test_grades_validas(cols_synth):
    num, cat = cols_synth
    cands = make_candidates(num, cat)
    for nome in ["logreg", "hist_gb", "random_forest", "lgbm"]:
        grade = param_distributions(nome)
        if nome in cands:
            assert grade, nome
            assert all(k.startswith("model__") for k in grade)
        else:
            assert grade == {} or nome == "lgbm"


def test_todos_candidatos_treinam_e_predizem(split_synth, cols_synth):
    num, cat = cols_synth
    for nome, pipe in make_candidates(num, cat).items():
        pipe.fit(split_synth.X_train, split_synth.y_train)
        p = pipe.predict_proba(split_synth.X_test)[:, 1]
        assert p.shape == (len(split_synth.y_test),)
        assert np.isfinite(p).all(), nome


# --------------------------------------------------------------------------
# Ablação
# --------------------------------------------------------------------------

def test_ablacao_remove_bloco_historico(split_synth):
    cfg = TrainConfig.fast()
    res = _ablacao_sem_historico("logreg", {}, split_synth, cfg)
    assert set(res) == {"cv", "holdout"}
    assert 0.0 <= res["holdout"]["roc_auc"] <= 1.0
    assert 0.0 <= res["cv"]["roc_auc"] <= 1.0


def test_ablacao_pior_ou_igual_que_completo(split_synth, cols_synth):
    """Com sinal no histórico (sintético), remover o bloco não pode melhorar
    de forma sistemática — toleramos empate por ruído de amostra pequena."""
    num, cat = cols_synth
    cfg = TrainConfig.fast()
    pipe = make_candidates(num, cat)["logreg"]
    pipe.fit(split_synth.X_train, split_synth.y_train)
    p_full = pipe.predict_proba(split_synth.X_test)[:, 1]
    from src.evaluation.metrics import evaluate_binary
    auc_full = evaluate_binary(split_synth.y_test, p_full, 0.5)["roc_auc"]
    ab = _ablacao_sem_historico("logreg", {}, split_synth, cfg)
    assert ab["holdout"]["roc_auc"] <= auc_full + 0.05


# --------------------------------------------------------------------------
# Fim a fim
# --------------------------------------------------------------------------

def test_treinar_fim_a_fim_sem_salvar(matriz_2025):
    cfg = TrainConfig.fast()
    cfg.salvar = False
    res = treinar(matriz_2025, cfg)
    assert res.best_name in {"logreg", "hist_gb", "random_forest", "lgbm"}
    assert isinstance(res.pipeline, Pipeline)
    m = res.holdout[res.best_name]
    assert 0.5 <= m["roc_auc"] <= 1.0          # sinal forte no sintético
    assert "pr_auc_risco" in m
    assert set(res.ablacao) == {"cv", "holdout"}
    assert res.n_features > 0


def test_treinar_reprodutivel(matriz_2025):
    cfg = TrainConfig.fast()
    cfg.salvar = False
    r1 = treinar(matriz_2025, cfg)
    r2 = treinar(matriz_2025, cfg)
    assert r1.best_name == r2.best_name
    assert r1.holdout[r1.best_name]["roc_auc"] == pytest.approx(
        r2.holdout[r2.best_name]["roc_auc"])


def test_roundtrip_joblib(tmp_path, split_synth, cols_synth):
    num, cat = cols_synth
    pipe = make_candidates(num, cat)["logreg"]
    pipe.fit(split_synth.X_train, split_synth.y_train)
    path = tmp_path / "m.joblib"
    joblib.dump({"pipeline": pipe}, path)
    loaded = joblib.load(path)["pipeline"]
    p = loaded.predict_proba(split_synth.X_test)[:, 1]
    assert np.isfinite(p).all()


# --------------------------------------------------------------------------
# Sanidade da projeção (predict_2026)
# --------------------------------------------------------------------------

def test_sanidade_ranking_coerente():
    rng = np.random.default_rng(SEED)
    n = 200
    esforco = rng.uniform(0, 30, n)
    ranking = pd.DataFrame({
        "p_nao_atingir_2026": 1 / (1 + np.exp(-(esforco - 15) / 4))
                              + rng.normal(0, 0.03, n),
        "esforco_2026": esforco,
        "atingiu_meta_2025": pd.array(
            (esforco < 15).astype(int), dtype="Int64"),
    })
    san = _sanidade(ranking)
    assert san["spearman_risco_esforco"] > 0.8
    assert san["risco_medio_falhou_2025"] > san["risco_medio_atingiu_2025"]
