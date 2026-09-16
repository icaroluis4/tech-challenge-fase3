# Interpretabilidade — Fase 7 (escopo reduzido, D14)

Permutation importance (queda de ROC-AUC no holdout) nos dois modelos + SHAP `LinearExplainer` no Modelo B. SHAP do Modelo A (HistGB, 400k) fica como evolução futura (D14).

## Modelo B — risco de não atingir a meta (logreg)

### Permutation importance (top-15)

| feature            |   importance_mean |   importance_std |
|:-------------------|------------------:|-----------------:|
| esforco_t          |            0.3141 |           0.0173 |
| total_mat_fund_ai  |            0.2014 |           0.0092 |
| sg_uf              |            0.1537 |           0.0097 |
| resultado_t1       |            0.0634 |           0.0075 |
| total_doc_fund_ai  |            0.0515 |           0.0073 |
| total_mat_1_2_ano  |            0.0421 |           0.0070 |
| regiao             |            0.0380 |           0.0024 |
| qtd_escolas        |            0.0297 |           0.0026 |
| delta_t1           |            0.0254 |           0.0017 |
| total_tur_fund_ai  |            0.0188 |           0.0036 |
| atingiu_t1         |            0.0169 |           0.0023 |
| meta_t             |            0.0105 |           0.0027 |
| renda_pc           |            0.0103 |           0.0024 |
| idhm               |            0.0083 |           0.0014 |
| pct_escolas_rurais |            0.0071 |           0.0019 |

### SHAP — |valor| médio (top-15, features já transformadas)

| feature           |   mean_abs_shap |
|:------------------|----------------:|
| esforco_t         |          1.6691 |
| total_mat_fund_ai |          1.5232 |
| resultado_t1      |          0.8315 |
| total_doc_fund_ai |          0.6616 |
| delta_t1          |          0.6087 |
| total_mat_1_2_ano |          0.5425 |
| qtd_escolas       |          0.4006 |
| atingiu_t1        |          0.3739 |
| renda_pc          |          0.3370 |
| total_tur_fund_ai |          0.3072 |
| sg_uf_RS          |          0.2829 |
| regiao_S          |          0.2648 |
| sg_uf_SP          |          0.2455 |
| regiao_NE         |          0.2225 |
| sg_uf_PR          |          0.2180 |

Beeswarm: `images/interp_03_shap_risco.png`. Leitura: valores SHAP positivos empurram para **atingir** a meta (classe 1); negativos, para **não atingir**.

## Modelo A — aluno alfabetizado (HistGB, amostra 20k do holdout)

### Permutation importance (top-15)

| feature                    |   importance_mean |   importance_std |
|:---------------------------|------------------:|-----------------:|
| pc_alfabetizado_2024       |            0.0356 |           0.0015 |
| sg_uf                      |            0.0059 |           0.0008 |
| tp_dependencia             |            0.0055 |           0.0009 |
| ambicao_2030               |            0.0043 |           0.0002 |
| pc_alfabetizado_2023       |            0.0020 |           0.0003 |
| regiao                     |            0.0017 |           0.0005 |
| taxa_analfabetismo_15_mais |            0.0015 |           0.0002 |
| meta_2027                  |            0.0014 |           0.0001 |
| idhm_l                     |            0.0007 |           0.0002 |
| qtd_escolas_estaduais      |            0.0007 |           0.0002 |
| share_va_agro              |            0.0007 |           0.0002 |
| meta_2026                  |            0.0006 |           0.0001 |
| meta_2028                  |            0.0005 |           0.0001 |
| capital_uf                 |            0.0005 |           0.0001 |
| idhm_e                     |            0.0005 |           0.0001 |

Figura: `images/interp_01_permutation_aluno.png`.

## Leitura cruzada (A × B)

As duas listas devem ser dominadas por resultado histórico do município (`pc_alfabetizado_*`), rede/estrutura escolar e socioeconomia (IDHM/PIB) — coerente com ICC≈0,08 (risco contextual) e com a ablação D13 (contexto sozinho prediz, AUC 0,79).
