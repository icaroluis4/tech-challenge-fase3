"""Pipeline de pré-processamento: fit em dados sintéticos com NaN, dtypes
nullable do pandas, roundtrip joblib e splits."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED, SEED
from src.preprocessing.leakage import assert_no_leakage
from src.preprocessing.splits import (
    group_cv, sample_aluno, split_aluno, split_municipio, stratified_cv,
)
from src.preprocessing.transformers import (
    build_preprocessor, get_feature_names, infer_feature_columns,
)

N = 1_000


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({
        "co_municipio": pd.array(rng.integers(1_100_000, 5_300_000, N), dtype="Int64"),
        "no_municipio": [f"m{i}" for i in range(N)],
        "pc_alfabetizado_2024": rng.uniform(20, 95, N),
        "meta_2025": rng.uniform(30, 90, N),
        "populacao_2024": pd.array(rng.integers(1_000, 2_000_000, N), dtype="Int64"),
        "qtd_escolas": pd.array(rng.integers(1, 300, N), dtype="Int64"),
        "idhm_e": rng.uniform(0.3, 0.9, N),
        "pct_escolas_rurais": pd.array(rng.uniform(0, 1, N), dtype="Float64"),
        "sg_uf": rng.choice(["SP", "BA", "CE", "AM", "RS"], N),
        "regiao": rng.choice(["N", "NE", "CO", "SE", "S"], N),
        "porte": pd.Categorical(rng.choice(["pequeno_1", "pequeno_2", "medio", "grande"], N)),
        "capital_uf": pd.array(rng.integers(0, 2, N), dtype="Int64"),
        "nome_regiao_intermediaria": rng.choice([f"ri{i}" for i in range(40)], N),
        "atingiu_meta_2025": pd.array(rng.integers(0, 2, N), dtype="boolean"),
        # leakage propositais — devem ser removidas pelo helper
        "pc_alfabetizado_aeeb_2025": rng.uniform(20, 95, N),
        "gap_meta_2025": rng.normal(0, 5, N),
    })
    # injeta NaN em todos os tipos
    for c in ["pc_alfabetizado_2024", "populacao_2024", "idhm_e", "pct_escolas_rurais",
              "sg_uf", "porte", "capital_uf", "atingiu_meta_2025"]:
        mask = rng.random(N) < 0.08
        df.loc[mask, c] = pd.NA if df[c].dtype != object else None
    return df


def _pipe(df, target="atingiu_meta_2025"):
    num, cat = infer_feature_columns(df, target=target)
    pre = build_preprocessor(num, cat, min_frequency=5)
    return num, cat, Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=500))])


def test_infer_remove_leakage_ids_e_descritivas(synth):
    num, cat = infer_feature_columns(synth, target="atingiu_meta_2025")
    assert_no_leakage(num + cat)
    assert "atingiu_meta_2025" not in num + cat
    assert "nome_regiao_intermediaria" not in num + cat
    assert set(cat) == {"sg_uf", "regiao", "porte", "capital_uf"}
    assert "populacao_2024" in num and "idhm_e" in num


def test_fit_predict_proba_com_nan(synth):
    d = synth.dropna(subset=["atingiu_meta_2025"])
    y = d["atingiu_meta_2025"].astype(int)
    _, _, pipe = _pipe(d)
    pipe.fit(d, y)
    proba = pipe.predict_proba(d)
    assert proba.shape == (len(d), 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sem_nan_apos_transform(synth):
    d = synth.dropna(subset=["atingiu_meta_2025"])
    num, cat, pipe = _pipe(d)
    Xt = pipe.named_steps["pre"].fit_transform(d)
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()
    names = get_feature_names(pipe.named_steps["pre"])
    assert Xt.shape[1] == len(names)
    assert any(n.startswith("sg_uf_") for n in names)


def test_categoria_desconhecida_no_predict(synth):
    d = synth.dropna(subset=["atingiu_meta_2025"])
    y = d["atingiu_meta_2025"].astype(int)
    _, _, pipe = _pipe(d)
    pipe.fit(d, y)
    novo = d.head(5).copy()
    novo["sg_uf"] = "ZZ"
    novo["porte"] = pd.Categorical(["metropole"] * 5)
    assert pipe.predict_proba(novo).shape == (5, 2)


def test_joblib_roundtrip(synth, tmp_path):
    d = synth.dropna(subset=["atingiu_meta_2025"])
    y = d["atingiu_meta_2025"].astype(int)
    _, _, pipe = _pipe(d)
    pipe.fit(d, y)
    p = tmp_path / "pipe.joblib"
    joblib.dump(pipe, p)
    loaded = joblib.load(p)
    assert np.allclose(loaded.predict_proba(d), pipe.predict_proba(d))


def test_scale_false_passthrough(synth):
    d = synth.dropna(subset=["atingiu_meta_2025"])
    num, cat = infer_feature_columns(d, target="atingiu_meta_2025")
    pre = build_preprocessor(num, cat, scale=False, min_frequency=5)
    Xt = pre.fit_transform(d)
    names = get_feature_names(pre)
    # sem scaler, log1p(populacao) deve estar na escala log (~7..14)
    col = names.index("populacao_2024")
    assert 5 < np.nanmedian(Xt[:, col]) < 16


# ---------------------------------------------------------------- splits

def test_split_municipio_sintetico(synth):
    sp = split_municipio(synth)
    n_valid = synth["atingiu_meta_2025"].notna().sum()
    assert len(sp.y_train) + len(sp.y_test) == n_valid
    assert abs(len(sp.y_test) / n_valid - 0.2) < 0.01
    assert abs(sp.y_train.mean() - sp.y_test.mean()) < 0.05
    assert "atingiu_meta_2025" not in sp.X_train.columns
    assert sp.groups_train is not None and len(sp.groups_train) == len(sp.y_train)


def test_split_aluno_sintetico(synth):
    d = synth.rename(columns={"atingiu_meta_2025": "in_alfabetizado"}).dropna(subset=["in_alfabetizado"])
    sp = split_aluno(d)
    assert set(sp.groups_train.unique()).issubset(set(d["co_municipio"].dropna().unique()))
    assert "in_alfabetizado" not in sp.X_test.columns


def test_group_cv_nao_vaza_grupo(synth):
    d = synth.dropna(subset=["atingiu_meta_2025"]).reset_index(drop=True)
    g = d["sg_uf"].fillna("NA")
    for tr, va in group_cv(5).split(d, groups=g):
        assert not set(g.iloc[tr]) & set(g.iloc[va])


def test_stratified_cv_reprodutivel(synth):
    d = synth.dropna(subset=["atingiu_meta_2025"]).reset_index(drop=True)
    y = d["atingiu_meta_2025"].astype(int)
    a = [va for _, va in stratified_cv().split(d, y)]
    b = [va for _, va in stratified_cv().split(d, y)]
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_sample_aluno_estratificada(synth):
    d = synth.rename(columns={"atingiu_meta_2025": "in_alfabetizado"}).dropna(subset=["in_alfabetizado", "sg_uf"])
    d["in_alfabetizado"] = d["in_alfabetizado"].astype(int)
    s = sample_aluno(d, n=300)
    assert abs(len(s) - 300) <= 15
    assert abs(s["in_alfabetizado"].mean() - d["in_alfabetizado"].mean()) < 0.06
    assert len(sample_aluno(d, n=10_000)) == len(d)


# ------------------------------------------------- dados reais (se existirem)

REAL = DATA_PROCESSED / "features_municipio.parquet"
TGT = DATA_PROCESSED / "targets_municipio.parquet"


@pytest.mark.skipif(not (REAL.exists() and TGT.exists()), reason="dados reais ausentes")
def test_pipeline_em_features_reais():
    f = pd.read_parquet(REAL)
    t = pd.read_parquet(TGT)[["co_municipio", "atingiu_meta_2025"]]
    d = f.merge(t, on="co_municipio", how="inner")
    sp = split_municipio(d)
    num, cat = infer_feature_columns(sp.X_train)
    assert_no_leakage(num + cat)
    pipe = Pipeline([("pre", build_preprocessor(num, cat)),
                     ("model", LogisticRegression(max_iter=2000))])
    pipe.fit(sp.X_train, sp.y_train)
    proba = pipe.predict_proba(sp.X_test)[:, 1]
    assert np.isfinite(proba).all()
    assert len(get_feature_names(pipe.named_steps["pre"])) > len(num)
