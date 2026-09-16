"""Modelo A (aluno): candidatos, grades de hiperparâmetros, agregação municipal,
curva de overfitting e execução fim a fim em dados sintéticos pequenos.

Os testes NÃO dependem do parquet real (gitignored) — usam um gerador sintético
que reproduz o desenho do problema: target individual com sinal apenas no
**contexto municipal** (ICC baixo, G3).
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.evaluation.metrics import evaluate_binary
from src.modeling.train_aluno import (
    CUSTO_FALSO_ALARME,
    CUSTO_MISS_RISCO,
    EXTRA_DROP,
    MIN_ALUNOS_AGG,
    TrainConfig,
    _correlacoes_municipais,
    agregado_municipal,
    carregar_dataset,
    learning_curve_boosting,
    make_candidates,
    param_distributions,
    treinar,
)
from src.preprocessing.leakage import assert_no_leakage
from src.preprocessing.splits import TARGET_ALUNO, split_aluno
from src.preprocessing.transformers import infer_feature_columns

N_MUN = 60
N_ALUNO = 40


@pytest.fixture(scope="module")
def aluno_synth() -> pd.DataFrame:
    """~2.400 alunos em 60 municípios; probabilidade individual depende só do
    contexto municipal (como no problema real)."""
    rng = np.random.default_rng(SEED)
    ufs = ["SP", "BA", "CE", "AM", "RS"]
    mun = pd.DataFrame({
        "co_municipio": pd.array(rng.choice(np.arange(1_100_000, 5_300_000), N_MUN,
                                            replace=False), dtype="Int64"),
        "no_municipio": [f"m{i}" for i in range(N_MUN)],
        "sg_uf": rng.choice(ufs, N_MUN),
        "co_uf": pd.array(rng.integers(11, 53, N_MUN), dtype="Int64"),
        "pc_alfabetizado_2023": rng.uniform(20, 90, N_MUN),
        "pc_alfabetizado_2024": rng.uniform(25, 95, N_MUN),
        "meta_2025": rng.uniform(30, 95, N_MUN),
        "meta_2026": rng.uniform(35, 97, N_MUN),
        "populacao_2024": pd.array(rng.integers(1_000, 3_000_000, N_MUN), dtype="Int64"),
        "qtd_escolas": pd.array(rng.integers(1, 400, N_MUN), dtype="Int64"),
        "total_mat_fund_ai": pd.array(rng.integers(50, 90_000, N_MUN), dtype="Int64"),
        "idhm_e": rng.uniform(0.3, 0.9, N_MUN),
        "renda_pc": rng.uniform(150, 2_200, N_MUN),
        "pct_escolas_rurais": pd.array(rng.uniform(0, 1, N_MUN), dtype="Float64"),
        "va_agropecuaria": rng.normal(2e5, 3e5, N_MUN),   # pode ser negativo (G10)
        "regiao": rng.choice(["N", "NE", "CO", "SE", "S"], N_MUN),
        "porte": pd.Categorical(rng.choice(["pequeno_1", "medio", "grande"], N_MUN)),
        "capital_uf": pd.array(rng.integers(0, 2, N_MUN), dtype="Int64"),
        "nome_regiao_intermediaria": rng.choice([f"ri{i}" for i in range(8)], N_MUN),
    })
    # nulos em vários dtypes (a imputação do pipeline tem de dar conta)
    for c in ["pc_alfabetizado_2023", "idhm_e", "porte", "capital_uf", "qtd_escolas"]:
        mun.loc[rng.random(N_MUN) < 0.1, c] = pd.NA

    mun["_p"] = 1 / (1 + np.exp(-(
        0.06 * (mun["pc_alfabetizado_2024"] - 60) + 2.5 * (mun["idhm_e"].fillna(0.6) - 0.6)
    )))
    df = mun.loc[mun.index.repeat(N_ALUNO)].reset_index(drop=True)
    df["id_aluno"] = np.arange(len(df))
    df["tp_dependencia"] = pd.array(rng.choice([2, 3], len(df)), dtype="Int64")
    df[TARGET_ALUNO] = pd.array(rng.binomial(1, df["_p"].to_numpy()), dtype="Int64")
    return df.drop(columns="_p")


@pytest.fixture(scope="module")
def split_synth(aluno_synth):
    return split_aluno(aluno_synth, seed=SEED)


@pytest.fixture(scope="module")
def cols_synth(split_synth):
    return infer_feature_columns(split_synth.X_train, extra_drop=EXTRA_DROP)


# --------------------------------------------------------------------------
# Features / leakage
# --------------------------------------------------------------------------

def test_features_sem_leakage_nem_ids(cols_synth):
    num, cat = cols_synth
    assert_no_leakage(num + cat)
    assert "id_aluno" not in num + cat
    assert "co_municipio" not in num + cat
    assert "co_uf" not in num + cat           # removido por EXTRA_DROP
    assert "sg_uf" in cat and "tp_dependencia" in cat


def test_target_nao_entra_como_feature(cols_synth):
    num, cat = cols_synth
    assert TARGET_ALUNO not in num + cat


# --------------------------------------------------------------------------
# Candidatos
# --------------------------------------------------------------------------

def test_candidatos_esperados(cols_synth):
    num, cat = cols_synth
    cands = make_candidates(num, cat)
    assert {"dummy", "logreg", "hist_gb", "random_forest"} <= set(cands)
    assert all(isinstance(p, Pipeline) for p in cands.values())
    assert all(list(p.named_steps) == ["pre", "model"] for p in cands.values())


def test_lineares_escalam_arvores_nao(cols_synth):
    num, cat = cols_synth
    cands = make_candidates(num, cat)

    def tem_scaler(pipe):
        pre = pipe.named_steps["pre"]
        return any(t[1].named_steps["scaler"] != "passthrough"
                   for t in pre.transformers if t[0] in {"num", "log"})

    assert tem_scaler(cands["logreg"])
    assert not tem_scaler(cands["hist_gb"])
    assert not tem_scaler(cands["random_forest"])


@pytest.mark.parametrize("nome", ["dummy", "logreg", "hist_gb", "random_forest"])
def test_cada_candidato_treina_e_prediz(nome, cols_synth, split_synth):
    num, cat = cols_synth
    pipe = make_candidates(num, cat)[nome]
    pipe.fit(split_synth.X_train, split_synth.y_train)
    p = pipe.predict_proba(split_synth.X_test)[:, 1]
    assert p.shape == split_synth.y_test.shape
    assert not np.isnan(p).any()
    assert p.min() >= 0 and p.max() <= 1


def test_dummy_fica_no_piso_da_prevalencia(cols_synth, split_synth):
    num, cat = cols_synth
    pipe = make_candidates(num, cat)["dummy"]
    pipe.fit(split_synth.X_train, split_synth.y_train)
    m = evaluate_binary(split_synth.y_test,
                        pipe.predict_proba(split_synth.X_test)[:, 1], 0.5)
    assert m["accuracy"] == pytest.approx(split_synth.y_test.mean(), abs=0.02)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.01)


def test_modelo_supera_dummy(cols_synth, split_synth):
    """Com sinal municipal injetado, o HistGB tem de bater o piso aleatório."""
    num, cat = cols_synth
    pipe = make_candidates(num, cat)["hist_gb"]
    pipe.fit(split_synth.X_train, split_synth.y_train)
    m = evaluate_binary(split_synth.y_test,
                        pipe.predict_proba(split_synth.X_test)[:, 1], 0.5)
    assert m["roc_auc"] > 0.55


def test_roundtrip_joblib(tmp_path, cols_synth, split_synth):
    num, cat = cols_synth
    pipe = make_candidates(num, cat)["logreg"].fit(split_synth.X_train, split_synth.y_train)
    esperado = pipe.predict_proba(split_synth.X_test)[:, 1]
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    obtido = joblib.load(path).predict_proba(split_synth.X_test)[:, 1]
    np.testing.assert_allclose(esperado, obtido)


# --------------------------------------------------------------------------
# Grades de hiperparâmetros
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nome", ["lgbm", "hist_gb", "random_forest", "logreg"])
def test_grade_prefixada_e_com_regularizacao(nome):
    grade = param_distributions(nome)
    assert grade, f"grade vazia para {nome}"
    assert all(k.startswith("model__") for k in grade)
    if nome != "logreg":
        assert any("min_child_samples" in k or "min_samples_leaf" in k for k in grade), \
            "grade deve conter controle explícito de overfitting"


def test_grade_valida_no_estimador(cols_synth):
    """Toda chave da grade precisa existir no `Pipeline` correspondente."""
    num, cat = cols_synth
    cands = make_candidates(num, cat)
    for nome, pipe in cands.items():
        grade = param_distributions(nome)
        validos = set(pipe.get_params(deep=True))
        assert set(grade) <= validos, f"params inválidos em {nome}: {set(grade) - validos}"


def test_grade_modelo_desconhecido_e_vazia():
    assert param_distributions("inexistente") == {}


# --------------------------------------------------------------------------
# Agregação municipal e curva de overfitting
# --------------------------------------------------------------------------

def test_agregado_municipal_estrutura_e_correlacao(cols_synth, split_synth):
    num, cat = cols_synth
    pipe = make_candidates(num, cat)["hist_gb"].fit(split_synth.X_train, split_synth.y_train)
    proba = pipe.predict_proba(split_synth.X_test)[:, 1]
    agg = agregado_municipal(split_synth.X_test, split_synth.y_test, proba)
    assert set(agg.columns) == {"co_municipio", "sg_uf", "n_alunos",
                                "taxa_observada", "taxa_prevista", "erro"}
    assert agg["n_alunos"].sum() == len(split_synth.y_test)
    assert agg["co_municipio"].is_unique
    assert agg["taxa_prevista"].between(0, 1).all()
    # a média das probabilidades por município deve acompanhar a taxa observada
    assert agg[["taxa_prevista", "taxa_observada"]].corr().iloc[0, 1] > 0.5


def test_correlacoes_municipais_reporta_subconjunto_com_massa():
    agg = pd.DataFrame({
        "co_municipio": range(10),
        "n_alunos": [5, 5, 5, 5, 5, 100, 100, 100, 100, 100],
        "taxa_observada": [0.1, 0.9, 0.2, 0.8, 0.5, 0.2, 0.4, 0.6, 0.8, 0.9],
        "taxa_prevista": [0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.45, 0.55, 0.75, 0.85],
    })
    agg["erro"] = agg["taxa_prevista"] - agg["taxa_observada"]
    c = _correlacoes_municipais(agg, min_alunos=MIN_ALUNOS_AGG)
    assert c["n_municipios"] == 10
    assert c["n_municipios_com_massa"] == 5
    # municípios com massa têm previsão quase perfeita; o ruído dos pequenos
    # derruba a correlação total
    assert c["corr_municipal_pearson_n30"] > 0.98
    assert c["corr_municipal_pearson_n30"] > c["corr_municipal_pearson"]
    assert c["mae_municipal"] == pytest.approx(agg["erro"].abs().mean())


def test_correlacoes_municipais_sem_massa_suficiente():
    agg = pd.DataFrame({"co_municipio": [1, 2], "n_alunos": [3, 4],
                        "taxa_observada": [0.2, 0.8], "taxa_prevista": [0.3, 0.7],
                        "erro": [0.1, -0.1]})
    c = _correlacoes_municipais(agg)
    assert np.isnan(c["corr_municipal_pearson_n30"])


def test_curva_overfit_gap_nao_negativo(cols_synth, split_synth):
    num, cat = cols_synth
    pipe = make_candidates(num, cat)["random_forest"]
    curva = learning_curve_boosting(pipe, split_synth, grid=(20, 60), seed=SEED)
    assert list(curva.columns) == ["n_estimators", "treino", "validacao"]
    assert len(curva) == 2
    # treino sempre ≥ validação (é o próprio conceito de gap de generalização)
    assert (curva["treino"] >= curva["validacao"] - 1e-9).all()


def test_curva_usa_max_iter_no_histgb(cols_synth, split_synth):
    num, cat = cols_synth
    curva = learning_curve_boosting(make_candidates(num, cat)["hist_gb"],
                                    split_synth, grid=(20, 50), seed=SEED)
    assert curva.columns[0] == "max_iter"


# --------------------------------------------------------------------------
# Constantes de decisão e execução fim a fim
# --------------------------------------------------------------------------

def test_custo_assimetrico_favorece_a_crianca():
    assert CUSTO_MISS_RISCO > CUSTO_FALSO_ALARME
    assert CUSTO_MISS_RISCO / CUSTO_FALSO_ALARME == pytest.approx(5.0)


def test_train_config_fast_e_menor():
    f, d = TrainConfig.fast(), TrainConfig()
    assert f.n_sample < d.n_sample and f.n_iter < d.n_iter
    assert f.cv_folds <= d.cv_folds


def test_treinar_fim_a_fim_sem_salvar(aluno_synth):
    cfg = TrainConfig(n_sample=0, cv_sample=0, cv_folds=3, n_iter=2, group_folds=3,
                      curva_grid=(20, 50), salvar=False, seed=SEED)
    res = treinar(aluno_synth, cfg)

    assert res.best_name in {"logreg", "hist_gb", "random_forest", "lgbm"}
    assert res.best_params and all(k.startswith("model__") for k in res.best_params)
    assert res.n_features > 10
    # baselines e os dois thresholds presentes no relatório de holdout
    assert {"dummy", "logreg", f"{res.best_name}@f1",
            f"{res.best_name}@custo5:1"} <= set(res.holdout)
    assert res.holdout[res.best_name]["roc_auc"] > res.holdout["dummy"]["roc_auc"]
    # thresholds válidos e custo finito
    assert 0 < res.thresholds["f1"] < 1
    assert 0 < res.thresholds["custo_5_1"] < 1
    assert res.thresholds["custo_medio_por_aluno"] > 0
    # CV estratificada de todos os candidatos + GroupKFold do melhor
    assert "dummy" in res.cv_strat and res.best_name in res.cv_strat
    assert "roc_auc" in res.cv_group
    assert res.cv_group["corr_municipal_pearson"] > 0
    assert res.municipal is not None and len(res.municipal) > 1
    assert res.duracao_s > 0
    # o pipeline devolvido prediz
    assert res.pipeline.predict_proba(aluno_synth.head(50).drop(columns=[TARGET_ALUNO])).shape \
        == (50, 2)


def test_treinar_amostragem_reprodutivel(aluno_synth):
    cfg = TrainConfig(n_sample=1_200, cv_sample=0, cv_folds=3, n_iter=2, group_folds=2,
                      curva_grid=(20,), salvar=False, seed=SEED)
    a = treinar(aluno_synth, cfg)
    b = treinar(aluno_synth, cfg)
    assert a.best_name == b.best_name
    assert a.holdout[a.best_name]["roc_auc"] == pytest.approx(
        b.holdout[b.best_name]["roc_auc"])


def test_carregar_dataset_arquivo_inexistente(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        carregar_dataset(tmp_path / "nao_existe.parquet")
