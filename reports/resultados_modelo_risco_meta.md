# Resultados — Modelo B (risco de não atingimento de meta)

> Gerado por `src/modeling/train_risco_meta.py` · seed=42 ·
> 5,417 municípios rotulados ·
> 46 features · duração 9 s.

## 1. Enquadramento

Target `atingiu_meta_2025` (1 = atingiu; prevalência 72.5%).
A classe de interesse é **não atingiu** (y=0) — métricas `*_risco`. Formulação
genérica em ``t``: o mesmo pipeline treina com t=2025 (backtest) e projeta
t=2026 (`predict_2026.py`). Colunas de ano explícito são dropadas e o bloco de
histórico é renomeado (`resultado_t1/t2`, `delta_t1`, `meta_t`, `esforco_t`,
`atingiu_t1`).

## 2. Divisão dos dados

- Holdout 80/20 estratificado: `train=4,333 (pos=0.725) | test=1,084 (pos=0.725)` (83 municípios sem rótulo
  descartados — G5).
- Validação cruzada `StratifiedKFold(3)` no treino.

## 3. Comparação de candidatos (StratifiedKFold)

| modelo | roc_auc | roc_auc_sd | pr_auc | f1 | balanced_accuracy | brier |
|---|---|---|---|---|---|---|
| dummy | 0.5000 | 0.0000 | 0.7249 | 0.8405 | 0.5000 | 0.2751 |
| logreg | 0.8151 | 0.0053 | 0.9086 | 0.8673 | 0.7064 | 0.1443 |
| hist_gb | 0.7948 | 0.0045 | 0.8987 | 0.8597 | 0.6893 | 0.1541 |
| random_forest | 0.8016 | 0.0075 | 0.9011 | 0.8657 | 0.6498 | 0.1507 |
| lgbm | 0.7995 | 0.0053 | 0.9010 | 0.8604 | 0.6875 | 0.1517 |

Melhor por ROC-AUC: **`logreg`**.

## 4. Hiperparâmetros (`RandomizedSearchCV`, n_iter=5)

```json
{
  "model__C": 12.32846739442066
}
```

## 5. Desempenho no holdout (backtest 2025)

| modelo | threshold | roc_auc | pr_auc | pr_auc_risco | f1 | f1_risco | balanced_accuracy | accuracy | recall_risco | precision_risco | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.5000 | 0.5000 | 0.7251 | 0.2749 | 0.8406 | 0.0000 | 0.5000 | 0.7251 | 0.0000 | 0.0000 | 0.2749 |
| logreg | 0.5000 | 0.8513 | 0.9306 | 0.7038 | 0.8736 | 0.6015 | 0.7208 | 0.8081 | 0.5268 | 0.7009 | 0.1328 |
| logreg@f1 | 0.4893 | 0.8513 | 0.9306 | 0.7038 | 0.8772 | 0.6058 | 0.7229 | 0.8127 | 0.5235 | 0.7189 | 0.1328 |
| logreg@custo5:1 | 0.7940 | 0.8513 | 0.9306 | 0.7038 | 0.7974 | 0.6438 | 0.7750 | 0.7417 | 0.8490 | 0.5184 | 0.1328 |

### 5.1 Thresholds

| threshold | valor | recall_risco | precision_risco | municípios sinalizados | leitura |
|---|---|---|---|---|---|
| padrão | 0.5000 | 0.527 | 0.701 | 224 | equilíbrio neutro |
| max F1 (classe atingiu) | 0.4893 | 0.523 | 0.719 | 217 | otimiza a classe majoritária |
| custo 5:1 (**adotado**) | 0.7940 | 0.849 | 0.518 | 488 | captura 85% dos que falhariam a meta |

### 5.2 Calibração

ECE (10 bins) = **0.0153** — `images/modelo_meta_02_calibracao.png`. Aqui as
probabilidades viram **ranking de prioridade** de gestão, então a calibração
importa mais que no Modelo A; se o ECE estiver alto, aplicar
`CalibratedClassifierCV` isotônico antes de citar probabilidades absolutas.

## 6. Ablação — valor incremental do contexto socioeconômico (D13)

| modelo | roc_auc_cv | roc_auc_holdout | pr_auc_risco_holdout | brier_holdout |
|---|---|---|---|---|
| logreg (completo) | 0.8151 | 0.8513 | 0.7038 | 0.1328 |
| logreg (sem histórico) | 0.7457 | 0.7900 | 0.6213 | 0.1528 |

O modelo "sem histórico" remove `resultado_t1/t2`, `delta_t1`, `meta_t`,
`esforco_t` e `atingiu_t1` — só restam Censo, IBGE, ADH e território. A queda
de AUC quantifica quanto do desempenho é **inércia do resultado passado**;
o que sobra é o valor preditivo do contexto socioeconômico.

## 7. Artefatos

- `models/modelo_risco_meta.joblib` (pipeline + thresholds)
- `images/modelo_meta_01_roc_pr.png` · `02_calibracao.png` · `03_confusao.png`
  · `04_candidatos.png`
- `reports/resultados_modelo_risco_meta.json` (métricas brutas)
- Projeção 2026: `python -m src.modeling.predict_2026` →
  `reports/ranking_risco_2026.csv`
