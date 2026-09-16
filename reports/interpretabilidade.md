# Interpretabilidade — Fase 7 (escopo reduzido, D14)

Permutation importance (queda de ROC-AUC no holdout) nos dois modelos + SHAP `LinearExplainer` no Modelo B. SHAP do Modelo A (HistGB, 400k) fica como evolução futura (D14).

## Modelo B — risco de não atingir a meta (logreg)

### Permutation importance (top-15)

| feature           |   importance_mean |   importance_std |
|:------------------|------------------:|-----------------:|
| esforco_t         |            0.3276 |           0.0172 |
| total_mat_fund_ai |            0.2397 |           0.0096 |
| sg_uf             |            0.1546 |           0.0097 |
| total_doc_fund_ai |            0.0723 |           0.0082 |
| resultado_t1      |            0.0693 |           0.0080 |
| total_mat_1_2_ano |            0.0615 |           0.0083 |
| regiao            |            0.0382 |           0.0024 |
| qtd_escolas       |            0.0321 |           0.0029 |
| delta_t1          |            0.0288 |           0.0017 |
| atingiu_t1        |            0.0167 |           0.0023 |
| idhm              |            0.0159 |           0.0021 |
| total_tur_fund_ai |            0.0121 |           0.0029 |
| renda_pc          |            0.0106 |           0.0024 |
| populacao_2024    |            0.0073 |           0.0026 |
| log_populacao     |            0.0073 |           0.0026 |

### SHAP — |valor| médio (top-15, features já transformadas)

| feature           |   mean_abs_shap |
|:------------------|----------------:|
| total_mat_fund_ai |          2.0197 |
| esforco_t         |          1.7968 |
| resultado_t1      |          0.9114 |
| total_doc_fund_ai |          0.8648 |
| total_mat_1_2_ano |          0.7162 |
| delta_t1          |          0.6794 |
| qtd_escolas       |          0.4210 |
| atingiu_t1        |          0.3735 |
| renda_pc          |          0.3467 |
| sg_uf_RS          |          0.2879 |
| regiao_S          |          0.2660 |
| total_tur_fund_ai |          0.2538 |
| sg_uf_SP          |          0.2506 |
| idhm              |          0.2458 |
| regiao_NE         |          0.2288 |

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
