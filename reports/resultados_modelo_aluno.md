# Resultados — Modelo A (aluno)

> Gerado por `src/modeling/train_aluno.py` · seed=42 ·
> amostra=399,999 linhas ·
> 55 features · duração 33.3 min.

## 1. Enquadramento

Target `in_alfabetizado` (1 = alfabetizado). Só alunos **presentes** (F2).
Nenhuma feature individual está disponível: escola anonimizada (F4) e todo dado
da prova é leakage determinístico (F1). O modelo observa **contexto municipal +
rede**, e o ICC municipal é ≈ 0,081 (G3) — ou seja, ~8% da variância individual
é explicável pelo município. **O teto de AUC é baixo por construção**; o produto
é um *score de risco contextual*, não um classificador individual.

## 2. Divisão dos dados

- Holdout 80/20 estratificado por target: `train=319,999 (pos=0.661) | test=80,000 (pos=0.661)`.
- Validação cruzada estratificada `StratifiedKFold(5)` em
  150,000 linhas do treino (custo computacional).
- Validação territorial `GroupKFold(5)` por `co_municipio`.

## 3. Comparação de candidatos (StratifiedKFold)

| modelo | roc_auc | roc_auc_sd | pr_auc | f1 | balanced_accuracy | brier |
|---|---|---|---|---|---|---|
| dummy | 0.5000 | 0.0000 | 0.6613 | 0.7961 | 0.5000 | 0.3387 |
| logreg | 0.6524 | 0.0014 | 0.7830 | 0.6538 | 0.6105 | 0.2324 |
| hist_gb | 0.6595 | 0.0008 | 0.7885 | 0.6617 | 0.6160 | 0.2304 |
| random_forest | 0.6594 | 0.0007 | 0.7885 | 0.6694 | 0.6150 | 0.2295 |
| lgbm | 0.6594 | 0.0006 | 0.7885 | 0.6626 | 0.6161 | 0.2304 |

Melhor por ROC-AUC: **`hist_gb`**.

## 4. Hiperparâmetros escolhidos (`RandomizedSearchCV`, n_iter=25)

```json
{
  "model__min_samples_leaf": 50,
  "model__max_leaf_nodes": 63,
  "model__max_iter": 300,
  "model__max_features": 0.6,
  "model__learning_rate": 0.02,
  "model__l2_regularization": 20.0
}
```

Controle de overfitting explícito na grade (`min_child_samples`/`min_samples_leaf`,
`reg_lambda`/`l2_regularization`, `num_leaves`/`max_leaf_nodes`, subamostragem de
colunas e linhas). Curva treino × validação em
`images/modelo_aluno_04_curva_overfit.png`: Validação máxima em `max_iter=200` (AUC=0.6619); gap treino−validação no último ponto = 0.0401. Além desse ponto o treino sobe e a validação cai — overfitting controlado pela regularização escolhida e pelo `max_iter` da busca.

## 5. Desempenho no holdout (nunca tocado no treino)

| modelo | threshold | roc_auc | pr_auc | pr_auc_risco | f1 | f1_risco | balanced_accuracy | accuracy | recall_risco | precision_risco | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.5000 | 0.5000 | 0.6615 | 0.3385 | 0.7963 | 0.0000 | 0.5000 | 0.6615 | 0.0000 | 0.0000 | 0.3385 |
| logreg | 0.5000 | 0.6547 | 0.7854 | 0.4684 | 0.6564 | 0.5209 | 0.6103 | 0.5998 | 0.6427 | 0.4380 | 0.2318 |
| hist_gb | 0.5000 | 0.6659 | 0.7947 | 0.4838 | 0.6634 | 0.5305 | 0.6193 | 0.6079 | 0.6544 | 0.4461 | 0.2289 |
| hist_gb@f1 | 0.2725 | 0.6659 | 0.7947 | 0.4838 | 0.7973 | 0.0713 | 0.5135 | 0.6672 | 0.0377 | 0.6448 | 0.2289 |
| hist_gb@custo5:1 | 0.7270 | 0.6659 | 0.7947 | 0.4838 | 0.2158 | 0.5279 | 0.5480 | 0.4106 | 0.9735 | 0.3622 | 0.2289 |

`pr_auc_risco`, `f1_risco`, `recall_risco` e `precision_risco` tratam a classe
**não alfabetizado** como positiva — é ela que orienta política pública.

### 5.1 Escolha do threshold (D9)

| threshold | valor | recall_risco | precision_risco | alunos sinalizados | leitura |
|---|---|---|---|---|---|
| padrão | 0.5000 | 0.654 | 0.446 | 39,729 | equilíbrio neutro |
| max F1 (classe alfabetizado) | 0.2725 | 0.038 | 0.645 | 1,585 | quase ninguém sinalizado — **inútil para política** |
| custo 5:1 (**adotado**) | 0.7270 | 0.973 | 0.362 | 72,794 | captura 97% dos não alfabetizados ao custo de ampla triagem |

O threshold de máximo F1 otimiza a classe majoritária e praticamente elimina a
sinalização de risco — o oposto do objetivo. O custo assimétrico 5:1 (perder uma
criança em risco pesa 5× um falso alarme) leva a um threshold alto
(0.727): o modelo passa a operar como **triagem ampla**,
com recall de risco de 97.3% e custo médio de
0.6253 por aluno. Como só há contexto municipal
(sem features individuais), esse é o uso legítimo: **priorizar territórios/redes**, não
rotular crianças.

### 5.2 Calibração

ECE (10 bins de quantil) = **0.1469** — `images/modelo_aluno_02_calibracao.png`.
Com `class_weight="balanced"` o modelo **subestima** sistematicamente a probabilidade
de alfabetização (curva acima da diagonal): as probabilidades são úteis para
**ordenar** risco, mas não devem ser lidas como frequências absolutas sem
recalibração (Platt/isotônica) — fora do escopo desta fase.

## 6. Generalização territorial (`GroupKFold` por município)

- **roc_auc**: 0.6473
- **pr_auc**: 0.7809
- **f1**: 0.6487
- **balanced_accuracy**: 0.6080
- **brier**: 0.2349
- **corr_municipal_pearson**: 0.5103
- **corr_municipal_spearman**: 0.5585
- **n_municipios**: 5124.0000
- **n_municipios_com_massa**: 486.0000
- **corr_municipal_pearson_n30**: 0.8858
- **mae_municipal**: 0.2197

Agregação municipal (média das probabilidades × taxa observada): 5,124 municípios no holdout · erro médio -0.1358 · MAE 0.2197 · Pearson 0.5103 (restrito a 486 municípios com n≥30 alunos amostrados: 0.8858).
Arquivo: `reports/modelo_aluno_agregado_municipal.csv`.

> A correlação total é deprimida por **erro binomial**: com poucos alunos
> amostrados por município, a taxa observada tem desvio ≈ 0,5/√n. A coluna
> restrita a n≥30 é a leitura correta da capacidade de ordenar
> municípios por risco.

## 7. Leitura dos resultados

- **AUC individual 0.666 (holdout) / 0.647 (GroupKFold)** —
  dentro da faixa esperada 0,62–0,72 dado o ICC≈0,08. A queda de
  0.019 entre
  as duas visões é o **custo de generalizar para municípios nunca vistos**; é o
  número que deve ser citado.
- Todos os modelos não triviais ficam a ≤0,01 de AUC entre si (§3): o limite é a
  **informação disponível**, não a classe de modelo. Mais complexidade não ajuda.
- **Correlação municipal 0.89** (n≥30):
  agregado, o modelo reproduz bem a taxa de alfabetização observada — confirma
  o enquadramento de *score de risco contextual*.
- Uso recomendado: ranquear municípios/redes por risco médio; **não** decidir
  intervenção individual com este modelo.

## 8. Artefatos

- `models/modelo_aluno.joblib` (pipeline completo + thresholds)
- `images/modelo_aluno_01_roc_pr.png` · `02_calibracao.png` · `03_confusao.png`
  · `04_curva_overfit.png` · `05_candidatos.png`
- `reports/resultados_modelo_aluno.json` (métricas brutas)
