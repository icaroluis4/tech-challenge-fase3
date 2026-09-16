# Resultados — Modelo B (risco de não atingimento de meta)

> Gerado por `src/modeling/train_risco_meta.py` · seed=42 ·
> 5,417 municípios rotulados ·
> 46 features · duração 16 s.

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
- Validação cruzada `StratifiedKFold(5)` no treino.

## 3. Comparação de candidatos (StratifiedKFold)

| modelo | roc_auc | roc_auc_sd | pr_auc | f1 | balanced_accuracy | brier |
|---|---|---|---|---|---|---|
| dummy | 0.5000 | 0.0000 | 0.7249 | 0.8405 | 0.5000 | 0.2751 |
| logreg | 0.8171 | 0.0074 | 0.9102 | 0.8677 | 0.7071 | 0.1433 |
| hist_gb | 0.7993 | 0.0026 | 0.9004 | 0.8605 | 0.6942 | 0.1516 |
| random_forest | 0.8046 | 0.0118 | 0.9027 | 0.8685 | 0.6602 | 0.1494 |
| lgbm | 0.7977 | 0.0038 | 0.8996 | 0.8598 | 0.6887 | 0.1519 |

Melhor por ROC-AUC: **`logreg`**.

## 4. Hiperparâmetros (`RandomizedSearchCV`, n_iter=25)

```json
{
  "model__C": 4.328761281083057
}
```

## 5. Desempenho no holdout (backtest 2025)

| modelo | threshold | roc_auc | pr_auc | pr_auc_risco | f1 | f1_risco | balanced_accuracy | accuracy | recall_risco | precision_risco | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.5000 | 0.5000 | 0.7251 | 0.2749 | 0.8406 | 0.0000 | 0.5000 | 0.7251 | 0.0000 | 0.0000 | 0.2749 |
| logreg | 0.5000 | 0.8519 | 0.9309 | 0.7042 | 0.8743 | 0.6027 | 0.7214 | 0.8090 | 0.5268 | 0.7040 | 0.1327 |
| logreg@f1 | 0.4704 | 0.8519 | 0.9309 | 0.7042 | 0.8762 | 0.5913 | 0.7137 | 0.8100 | 0.5000 | 0.7233 | 0.1327 |
| logreg@custo5:1 | 0.8142 | 0.8519 | 0.9309 | 0.7042 | 0.7763 | 0.6308 | 0.7662 | 0.7214 | 0.8658 | 0.4962 | 0.1327 |

### 5.1 Thresholds

| threshold | valor | recall_risco | precision_risco | municípios sinalizados | leitura |
|---|---|---|---|---|---|
| padrão | 0.5000 | 0.527 | 0.704 | 223 | equilíbrio neutro |
| max F1 (classe atingiu) | 0.4704 | 0.500 | 0.723 | 206 | otimiza a classe majoritária |
| custo 5:1 (**adotado**) | 0.8142 | 0.866 | 0.496 | 520 | captura 87% dos que falhariam a meta |

### 5.2 Calibração

ECE (10 bins) = **0.0176** — `images/modelo_meta_02_calibracao.png`. Aqui as
probabilidades viram **ranking de prioridade** de gestão, então a calibração
importa mais que no Modelo A; se o ECE estiver alto, aplicar
`CalibratedClassifierCV` isotônico antes de citar probabilidades absolutas.

## 6. Ablação — valor incremental do contexto socioeconômico (D13)

| modelo | roc_auc_cv | roc_auc_holdout | pr_auc_risco_holdout | brier_holdout |
|---|---|---|---|---|
| logreg (completo) | 0.8171 | 0.8519 | 0.7042 | 0.1327 |
| logreg (sem histórico) | 0.7487 | 0.7904 | 0.6232 | 0.1526 |

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
