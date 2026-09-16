"""Métricas de avaliação: casos determinísticos, espelho na classe de risco,
thresholds (F1 e custo assimétrico), calibração e relatório de CV."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import SEED
from src.evaluation.metrics import (
    best_threshold_cost,
    best_threshold_f1,
    calibration_table,
    cv_report,
    evaluate_binary,
    expected_calibration_error,
    metrics_frame,
    to_markdown,
)
from src.preprocessing.splits import stratified_cv

N = 2_000


@pytest.fixture(scope="module")
def dados():
    """Sinal fraco e desbalanceamento parecidos com o do Modelo A (66% positivos)."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N, 3))
    logit = 0.7 + 0.8 * x[:, 0] - 0.4 * x[:, 1]
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    ruido = np.clip(p + rng.normal(0, 0.12, N), 0.001, 0.999)
    return pd.DataFrame(x, columns=["a", "b", "c"]), pd.Series(y), ruido


def test_classificador_perfeito():
    y = [0, 0, 1, 1]
    p = [0.01, 0.10, 0.90, 0.99]
    m = evaluate_binary(y, p, 0.5)
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["f1_risco"] == 1.0
    assert (m["fp"], m["fn"]) == (0, 0)
    assert m["brier"] < 0.01


def test_classificador_invertido_tem_auc_zero():
    m = evaluate_binary([0, 0, 1, 1], [0.99, 0.90, 0.10, 0.01], 0.5)
    assert m["roc_auc"] == 0.0


def test_matriz_confusao_e_linguagem_de_negocio():
    # y=0 previsto 1 → miss_risco (fp); y=1 previsto 0 → falso_alarme (fn)
    y = [0, 0, 0, 1, 1, 1]
    p = [0.9, 0.1, 0.2, 0.8, 0.9, 0.2]
    m = evaluate_binary(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["miss_risco"] == m["fp"] == 1
    assert m["falso_alarme"] == m["fn"] == 1
    assert m["n"] == 6
    assert m["prevalencia"] == pytest.approx(0.5)


def test_metricas_de_risco_sao_espelho(dados):
    _, y, p = dados
    m = evaluate_binary(y, p, 0.5)
    # PR-AUC da classe de risco usa (1-y, 1-p) → deve ficar entre 0 e 1 e
    # ser diferente da PR-AUC positiva (classes desbalanceadas)
    assert 0 < m["pr_auc_risco"] < 1
    assert m["pr_auc"] != pytest.approx(m["pr_auc_risco"])
    # recall_risco = tn / (tn + fp)
    assert m["recall_risco"] == pytest.approx(m["tn"] / (m["tn"] + m["fp"]))
    assert m["precision_risco"] == pytest.approx(m["tn"] / (m["tn"] + m["fn"]))


def test_evaluate_binary_rejeita_entrada_invalida():
    with pytest.raises(ValueError):
        evaluate_binary([0, 1], [0.5])                 # shapes diferentes
    with pytest.raises(ValueError):
        evaluate_binary([0, 1], [0.5, np.nan])         # NaN
    with pytest.raises(ValueError):
        evaluate_binary([0, 1], [0.5, 1.7])            # fora de [0, 1]


def test_classe_unica_nao_quebra():
    m = evaluate_binary([1, 1, 1], [0.6, 0.7, 0.8], 0.5)
    assert np.isnan(m["roc_auc"])
    assert m["accuracy"] == 1.0


def test_best_threshold_f1_maximiza(dados):
    from sklearn.metrics import f1_score
    _, y, p = dados
    t = best_threshold_f1(y, p)
    f1_otimo = f1_score(y, (p >= t).astype(int))
    for alt in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert f1_otimo >= f1_score(y, (p >= alt).astype(int)) - 1e-9


def test_threshold_custo_penaliza_miss_risco(dados):
    _, y, p = dados
    t_sim, custo_sim = best_threshold_cost(y, p, 1.0, 1.0)
    t_asym, _ = best_threshold_cost(y, p, 20.0, 1.0)
    # custo alto para deixar criança em risco passar ⇒ threshold mais ALTO
    # (mais alunos classificados como risco)
    assert t_asym >= t_sim
    m_asym = evaluate_binary(y, p, t_asym)
    m_sim = evaluate_binary(y, p, t_sim)
    assert m_asym["recall_risco"] >= m_sim["recall_risco"]
    assert custo_sim >= 0


def test_threshold_custo_rejeita_custo_invalido(dados):
    _, y, p = dados
    with pytest.raises(ValueError):
        best_threshold_cost(y, p, 0.0, 1.0)


def test_calibration_table_bem_calibrada(dados):
    _, y, p = dados
    tab = calibration_table(y, p, bins=10)
    assert tab["n"].sum() == len(y)
    assert tab["p_media"].is_monotonic_increasing
    # probas geradas do próprio DGP → boa calibração
    assert tab.attrs["ece"] < 0.05
    assert expected_calibration_error(y, p, bins=10) == pytest.approx(tab.attrs["ece"])


def test_calibration_table_detecta_descalibragem(dados):
    _, y, p = dados
    ruim = np.clip(p * 0.4, 0, 1)  # subestima sistematicamente
    assert expected_calibration_error(y, ruim) > expected_calibration_error(y, p)


def test_calibration_table_strategy_invalida(dados):
    _, y, p = dados
    with pytest.raises(ValueError):
        calibration_table(y, p, strategy="qualquer")


def test_cv_report_medias_e_desvios(dados):
    x, y, _ = dados
    res = cv_report(LogisticRegression(max_iter=500), x, y,
                    cv=stratified_cv(3, SEED), label="logreg")
    assert set(res.mean) == {"roc_auc", "pr_auc", "f1", "balanced_accuracy", "brier"}
    assert res.folds.shape[0] == 3
    assert 0.5 < res.mean["roc_auc"] < 1.0
    assert 0 < res.mean["brier"] < 0.5          # neg_brier já foi invertido
    assert all(v >= 0 for v in res.std.values())
    assert "logreg" in res.summary()
    assert res.to_frame().shape == (5, 3)


def test_cv_report_reprodutivel(dados):
    x, y, _ = dados
    kw = dict(cv=stratified_cv(3, SEED))
    a = cv_report(LogisticRegression(max_iter=500), x, y, **kw)
    b = cv_report(LogisticRegression(max_iter=500), x, y, **kw)
    assert a.mean == b.mean


def test_cv_report_dummy_fica_no_piso(dados):
    x, y, _ = dados
    res = cv_report(DummyClassifier(strategy="most_frequent", random_state=SEED),
                    x, y, cv=stratified_cv(3, SEED))
    assert res.mean["roc_auc"] == pytest.approx(0.5, abs=0.01)
    assert res.mean["balanced_accuracy"] == pytest.approx(0.5, abs=0.01)


def test_metrics_frame_e_markdown(dados):
    _, y, p = dados
    tabela = metrics_frame({"modelo_a": evaluate_binary(y, p, 0.5),
                            "modelo_b": evaluate_binary(y, 1 - p, 0.5)})
    assert list(tabela.index) == ["modelo_a", "modelo_b"]
    assert tabela.columns[0] == "roc_auc"       # ordem canônica respeitada
    md = to_markdown(tabela)
    assert md.startswith("|") and "roc_auc" in md
    assert len(md.splitlines()) == 4            # header + sep + 2 linhas


def test_to_markdown_trata_nan():
    df = pd.DataFrame({"x": [float("nan"), 1.0]}, index=["a", "b"])
    assert "—" in to_markdown(df)


def test_auc_consistente_com_sklearn(dados):
    _, y, p = dados
    assert evaluate_binary(y, p, 0.5)["roc_auc"] == pytest.approx(roc_auc_score(y, p))
