# PROGRESS — Tech Challenge Fase 3

> **Este arquivo é o ponto de retomada entre sessões.** Leia-o inteiro antes de
> continuar. O plano completo está em `../PLANO_IMPLEMENTACAO.md` (pasta `Desafio3/`).
> Atualize este arquivo ao fim de cada fase.

Última atualização: 2026-09-15 — **Fases 0–4 concluídas. Próxima: Fase 5 (Modelo B — risco de meta + projeção 2026).**

---

## 1. Onde estamos

| Fase | Descrição | Status |
|------|-----------|--------|
| 0 | Setup (estrutura, venv 3.11, git, repo remoto) | ✅ concluída |
| 1 | Extração BQ + feature store municipal | ✅ concluída (PR #1) |
| 2 | EDA municipal e por aluno | ✅ concluída (PR #2) |
| 3 | Pipeline de pré-processamento (`transformers.py`, `splits.py`) + testes | ✅ concluída (PR #3) |
| 4 | Modelo A — aluno (`train_aluno.py`, `metrics.py`) + relatório | ✅ concluída (PR #4) |
| **5** | **Modelo B — risco de meta + projeção 2026** | ⏭️ **PRÓXIMA** |
| 6 | Clusterização de municípios | ⬜ |
| 7 | Interpretabilidade (permutation importance + SHAP) | ⬜ |
| 8 | README final, decisões, roteiro do vídeo, `run_all.py` | ⬜ |

Git: `main` após PR #4 (ver `git log`). 4 PRs mergeados (squash). Repo:
`https://github.com/icaroluis4/tech-challenge-fase3` (privado).

---

## 2. Ambiente — comandos exatos

```powershell
cd "c:\Users\icaro\OneDrive\Área de Trabalho\Projetos\FIAP\Desafio3\tech-challenge-fase3"
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m src.data.build_features
.\.venv\Scripts\python.exe -m jupyter execute --inplace notebooks/01_eda_municipio.ipynb
.\.venv\Scripts\python.exe -m src.modeling.train_aluno --fast --no-save   # smoke (~1 min)
.\.venv\Scripts\python.exe -m src.modeling.train_aluno                    # completo (~33 min)
```

- Rodar scripts soltos que importam `src` exige `$env:PYTHONPATH="."` (ou usar `-m`).
- Log longo de treino: `... 2>&1 | Tee-Object -FilePath reports\_x.txt` cria UTF-16 —
  ler com `Get-Content`, não com ferramentas que esperam UTF-8. `reports/_*.txt` é gitignored.

- venv: **Python 3.11.8** em `.venv` (sklearn 1.9.1, lightgbm 4.7.0, shap 0.51.0, pyarrow 25.0.1).
- Kernel Jupyter registrado: **`fase3_venv`** ("Python 3.11 (fase3)"). Os dois
  notebooks já apontam para ele — o kernel `python3` global **não** funciona
  (pyarrow antigo → `Repetition level histogram size mismatch`).
- GCP: projeto `semiotic-primer-366516`, ADC ok. CLI `bq` **não** funciona → só Python SDK.
- Custo: só BigQuery (tabelas comuns, dentro do free tier). **Nenhum serviço
  pago foi ativado** — "feature store" no plano é só nome de artefato
  (parquet local + tabela `gold_ml.features_municipio`).

### Armadilhas do PowerShell (já sofridas)
- Nunca SQL com crases em `python -c "..."` → backtick é escape. Use arquivo `.py`.
- `python -c` com strings contendo `\"` quebra o parser → criar script temporário.
- **`Set-Content` corrompe UTF-8** de `.ipynb` → editar notebook só via script Python
  com `io.open(..., encoding='utf-8')` + `json.dump(..., ensure_ascii=False)`.
- Saídas de log em stderr fazem o PowerShell reportar exit code 1 mesmo com sucesso —
  verificar o conteúdo, não só o código de saída.

---

## 3. Artefatos existentes

### Código
| Arquivo | Conteúdo |
|---------|----------|
| `src/config.py` | `GCP_PROJECT_ID`, `TBL_*`, paths (`DATA_RAW`, `DATA_PROCESSED`, `MODELS_DIR`, `REPORTS_DIR`, `IMAGES_DIR`), `SEED=42`. Cria os diretórios no import. |
| `src/data/extract_bq.py` | `extract_gold_municipio`, `extract_gold_uf`, `extract_externos`, `extract_aluno`, `evidencia_leakage_proficiencia`, `run_all`. **Idempotente** (cache por parquet). |
| `src/data/build_features.py` | `build_features_municipio`, `build_dataset_aluno`, `materializar_gold_ml`, `run_all`. |
| `src/preprocessing/leakage.py` | `LEAKAGE_ALUNO`, `LEAKAGE_2025`, `IDS`, `BLACKLIST`, `assert_no_leakage(columns, allow=None)`. |
| `src/preprocessing/transformers.py` | `infer_feature_columns(df, target, extra_drop)` → `(num_cols, cat_cols)` já sem BLACKLIST/descritivas; `build_preprocessor(num, cat, scale=True, min_frequency=20)` → `ColumnTransformer` com ramos `log` (mediana→symlog1p→scaler), `num` (mediana→scaler), `cat` (moda→OHE dense, `handle_unknown="ignore"`); `get_feature_names(pre)`. Constantes: `CATEGORICAL_COLS`, `LOG1P_COLS`, `DESCRIPTIVE_COLS`. |
| `src/preprocessing/splits.py` | `Split` dataclass (`X_train/X_test/y_train/y_test/groups_*`, `.summary()`); `split_aluno(df)`, `split_municipio(df)` (dropa nulos do target), `stratified_cv()`, `group_cv()`, `sample_aluno(df, n=400_000)` estratificada por `(sg_uf, in_alfabetizado)`. |
| `src/visualization/plots.py` | tema seaborn + `save_fig(fig, path)` (backend Agg); `plot_roc_pr`, `plot_calibration`, `plot_confusion`, `plot_learning_curve`, `plot_metric_bars`. |
| `src/evaluation/metrics.py` | `evaluate_binary(y, proba, threshold)` → dict (ROC/PR-AUC, F1, bal. acc, Brier, log_loss, matriz + versões `_risco` com classe 0 positiva, `miss_risco`/`falso_alarme`); `best_threshold_f1`, `best_threshold_cost(y, p, c_fn, c_fp)`, `calibration_table` (attrs `ece`), `expected_calibration_error`, `cv_report(pipe, X, y, cv, groups)` → `CVResult(mean, std, summary())`, `metrics_frame`, `to_markdown`. |
| `src/modeling/train_aluno.py` | Fase 4 inteira: `carregar_dataset`, `make_candidates` (dummy/logreg/hist_gb/random_forest/lgbm), `param_distributions`, `learning_curve_boosting`, `agregado_municipal`, `treinar(df, TrainConfig)` → `TrainResult`, `_salvar_artefatos`, `_escrever_relatorio`. CLI `--fast`, `--no-save`, `--n-sample`, `--cv-sample`, `--n-iter`. Constantes `EXTRA_DROP=("co_uf",)`, `CUSTO_MISS_RISCO=5`, `MIN_ALUNOS_AGG=30`. |
| `tests/test_leakage.py` | 48 casos parametrizados. Verdes. |
| `tests/test_features.py` | grão único, 5.500 linhas, cobertura externos ≥98%, derivadas coerentes, **nenhuma coluna 100% nula, escala do PIB per capita**. Verdes. |
| `tests/test_pipeline.py` | 12 testes: fit/predict_proba com NaN em dtypes nullable, sem NaN pós-transform, categoria desconhecida, roundtrip joblib, `scale=False`, splits, GroupKFold sem vazamento, CV reprodutível, amostragem, pipeline em features reais. Verdes. |
| `tests/test_metrics.py` | 18 testes: métricas perfeitas/aleatórias, espelhamento de classe, thresholds F1 e custo, calibração/ECE, `cv_report` com e sem grupos, markdown. Verdes. |
| `tests/test_train_aluno.py` | 27 testes: features sem leakage, todos os candidatos treinam/predizem, dummy = prevalência, roundtrip joblib, grades válidas, gap da curva ≥0, agregação municipal, `treinar()` fim a fim sem salvar, reprodutibilidade, correlações municipais com/sem massa. Verdes. |

**Total: 109 testes verdes** (`pytest tests -q`, ~20 s).

### Modelo A — resultado (treino completo, 400k, 33 min)
| Métrica | Valor |
|---|---|
| Melhor candidato (StratifiedKFold 5, 150k) | `hist_gb` AUC 0,6595 (lgbm/RF 0,6594, logreg 0,6524, dummy 0,50) |
| Hiperparâmetros | `max_iter=300, learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=50, l2=20, max_features=0.6` |
| **Holdout 20% (80k)** | **ROC-AUC 0,666** · PR-AUC 0,795 · PR-AUC risco 0,484 · Brier 0,229 |
| **GroupKFold por município** | **ROC-AUC 0,647** (número honesto) |
| Threshold custo 5:1 (adotado, D9) | 0,727 → recall_risco 97,3 %, precision_risco 36,2 % |
| Threshold max-F1 | 0,2725 → recall_risco 3,8 % (inútil — documentado) |
| Calibração | ECE 0,147; `class_weight=balanced` subestima P(alfabetizado) → só ordenar, não ler como frequência |
| Curva overfit | validação máxima em `max_iter=200`; gap final 0,040 |
| Agregado municipal (holdout) | Pearson 0,51 (5.124 mun.) · **0,886 em 486 mun. com n≥30** · MAE 0,22 |

Modelos não triviais ficam a ≤0,01 de AUC entre si → o limite é a informação
(ICC≈0,08), não a classe de modelo. Enquadramento oficial: *score de risco contextual*.

### Dados (gitignored — regenerar com os comandos acima)
| Arquivo | Linhas | Cols |
|---------|--------|------|
| `data/raw/gold_municipio.parquet` | 5.500 | 47 |
| `data/raw/gold_uf.parquet` | 27 | 40 |
| `data/raw/externos_municipio.parquet` | 5.500 | 18 |
| `data/raw/aluno_presente.parquet` | 1.969.921 | 6 |
| `data/processed/features_municipio.parquet` | 5.500 | **59** |
| `data/processed/targets_municipio.parquet` | 5.500 | 4 |
| `data/processed/dataset_aluno.parquet` | 1.943.034 | **62** |

Também materializado no BQ: `semiotic-primer-366516.gold_ml.features_municipio`.

`models/modelo_aluno.joblib` (2,2 MB, gitignored) — dict `{pipeline, best_name,
best_params, thresholds, target, seed}`. Regenerar com `python -m src.modeling.train_aluno`.

### Docs/relatórios
- `docs/decisoes_analiticas.md` — **D1..D12** escritas (D8 pipeline+2 validações,
  D9 custo 5:1, D10 métricas espelhadas na classe de risco, D11 correlação municipal
  com corte n≥30, D12 amostra 400k).
- `reports/eda_municipio.md` — achados + hipóteses **H1–H5**.
- `reports/evidencia_leakage_proficiencia.md` — tabela min/max de proficiência por classe.
- `reports/resultados_modelo_aluno.md` (+ `.json` com métricas brutas,
  `modelo_aluno_agregado_municipal.csv`) — relatório completo do Modelo A (8 seções).
- `images/` — 16 arquivos: `eda_01..08`, `eda_aluno_01..03`, `modelo_aluno_01..05`
  (ROC/PR, calibração, confusão, curva overfit, candidatos).

---

## 4. Fatos de dados confirmados nesta sessão (não redescobrir)

| # | Fato |
|---|------|
| G1 | Cobertura externos: população 100%, PIB 100%, IDHM 99,9%, diretórios 100%. |
| G2 | Join aluno ↔ features: 1.969.921 → **1.943.034** (descartados **1,36%**). |
| G3 | **ICC municipal ≈ 0,081** → só ~8% da variância do target individual está entre municípios. Teto do Modelo A é baixo *por construção*; enquadrar como risco contextual. |
| G4 | Média municipal de alfabetizados 2025 por região: N 62,5 · NE 69,6 · S 72,8 · SE 73,5 · CO 81,2. CO teve o maior salto 2023→2025 (~16,7 p.p.). |
| G5 | Taxa de atingimento da meta 2025 = **72,5%** (3.927 de 5.417 rotulados; 83 nulos). |
| G6 | Leakage F1 confirmado visualmente: separação perfeita em proficiência 743. |
| G7 | `dataset_aluno` **não** tem sufixos `_x/_y` — `co_uf`/`sg_uf` do aluno são dropados antes do merge (fix aplicado em `build_dataset_aluno`). |
| G8 | **VA setorial** (`va`, `va_agropecuaria`, …) só existe até **2021** na Base dos Dados → extraído de 2021 (PIB continua 2023). Antes estava 100% nulo (D7). |
| G9 | `pib` da fonte já está em **R$** (não mil R$): `pib_per_capita_2023 = pib/pop` → mediana R$ 28,9 mil. Bug de `×1000` corrigido (D7). |
| G10 | `va_agropecuaria` é **negativo** em ~1% dos municípios → `share_va_agro` < 0; o ramo `log` usa `symlog1p` para não gerar NaN. |
| G11 | `capital_uf` é 0/1 (`Int64`) mas tratado como **categórica**; `tp_dependencia` só tem valores 2 (estadual) e 3 (municipal) entre presentes. |
| G12 | **Modelo A: AUC 0,666 holdout / 0,647 GroupKFold**, todos os não triviais dentro de 0,01 → teto informacional confirmado (ver G3). Não vale gastar mais tempo em tuning. |
| G13 | Em 60k linhas (`--fast`) a logreg vence; em 150k+ o boosting passa à frente por ~0,007. Efeito de tamanho de amostra, não de modelo. |
| G14 | `infer_feature_columns(extra_drop=("co_uf",))` → **50 numéricas + 5 categóricas** no dataset_aluno. |
| G15 | Threshold max-F1 na classe 1 (0,27) sinaliza <4 % dos não alfabetizados — F1 na classe majoritária é métrica errada para este problema; sempre usar `*_risco` ou custo. |
| G16 | Média de proba por município correlaciona 0,886 com a taxa observada quando n≥30 alunos no holdout; sem o corte, erro binomial derruba para 0,51. |

### Nomes de colunas (importante!)
- `features_municipio` / `targets_municipio`: **snake_case minúsculo** (`co_municipio`, `sg_uf`, `pc_alfabetizado_2024`, `regiao`, `porte`, `idhm_e`, …).
- `dataset_aluno`: **tudo minúsculo** (`id_aluno`, `co_municipio`, `tp_dependencia`, `in_alfabetizado`) + todas as colunas da feature store.
- `data/raw/aluno_presente.parquet` ainda está em **MAIÚSCULO** (`ID_ALUNO`, `CO_MUNICIPIO`, `IN_ALFABETIZADO`); o lowercase acontece em `build_dataset_aluno`.

### Dtypes que exigem cuidado
- `regiao` é `ArrowStringArray` (dtype `str`); `porte` é `category` **com NaN**.
- `atingiu_meta_2025` é **boolean nullable** → `sns.boxplot`/`np.sort` levantam
  `TypeError: boolean value of NA is ambiguous`. **Sempre** `dropna(subset=[...]).astype(int)`
  antes de plotar/agrupar. Já corrigido na célula 17 do notebook 01.
- Em `groupby` com categórica, usar `observed=True`.

### Features derivadas já criadas em `features_municipio`
`delta_2023_2024`, `gap_meta_2024`, `atingiu_meta_2024`, `esforco_2025`,
`ambicao_2030`, `pct_escolas_rurais`, `pct_escolas_privadas`,
`alunos_por_turma_ai`, `regiao`, `log_populacao`, `porte`,
`pib_per_capita_2023`, `share_va_agro`.

---

## 5. Próximo passo — Fase 5: Modelo B (risco de meta) — detalhado

Branch: `feature/modelo-risco-meta`. Arquivos: `src/modeling/train_risco_meta.py`,
`src/modeling/predict_2026.py`. Plano completo em `../PLANO_IMPLEMENTACAO.md` §Fase 5.

Reaproveitar da Fase 4 (**não reescrever**):
```python
from src.evaluation.metrics import evaluate_binary, cv_report, best_threshold_cost, calibration_table, metrics_frame, to_markdown
from src.visualization.plots import plot_roc_pr, plot_calibration, plot_confusion, plot_metric_bars, save_fig
from src.preprocessing.splits import split_municipio, stratified_cv
from src.preprocessing.transformers import infer_feature_columns, build_preprocessor
```
`evaluate_binary` já devolve `*_risco` (classe 0 = não atingiu como positiva) — é o
que a Fase 5 precisa. `train_aluno.py` serve de gabarito estrutural
(`TrainConfig`/`TrainResult`, `_comparar_candidatos`, `_buscar_hiperparametros`,
`_salvar_artefatos`, `_escrever_relatorio`).

1. **Renomear para o esquema genérico em t** (função `montar_matriz_t(features, targets, t)`):
   `resultado_t1`, `resultado_t2`, `delta_t1`, `meta_t`, `esforco_t`, `atingiu_t1`;
   demais features (Censo/IBGE/ADH/território) estáticas. **Dropar** colunas de ano
   explícito (`pc_alfabetizado_2023/2024`, `meta_2025/2026`, `delta_2023_2024`,
   `gap_meta_2024`, `esforco_2025`, `ambicao_2030`, `atingiu_meta_2024`) para não
   duplicar informação nem vazar 2025 — conferir com `assert_no_leakage`.
2. Treino t=2025: `X = montar_matriz_t(..., 2025)`, `y = atingiu_meta_2025` de
   `targets_municipio.parquet` (5.417 rotulados, 72,5 % positivos — G5).
   `split_municipio` (dropa nulos) + `StratifiedKFold(5)`.
3. Candidatos: dummy, `LogisticRegression(balanced, scale=True)`, `HistGB`, `RF`, `LGBM`.
   `RandomizedSearchCV(n_iter=25, scoring="roc_auc")`. Base pequena → treino rápido,
   pode usar `RepeatedStratifiedKFold` para reduzir variância.
4. **Ablação obrigatória** (risco "fácil demais" por inércia de `resultado_t1`):
   treinar também **sem histórico** (só contexto socioeconômico) e reportar as duas
   AUCs lado a lado. Mostra o valor incremental do socioeconômico.
5. Métricas: ROC-AUC, **PR-AUC da classe não-atingiu** (`pr_auc_risco`), Brier,
   calibração (se ECE alto, `CalibratedClassifierCV(method="isotonic")` — aqui as
   probabilidades viram ranking de prioridade, então calibração importa mais que na Fase 4).
6. `predict_2026.py`: `montar_matriz_t(..., 2026)` com `resultado_t1 =
   pc_alfabetizado_aeeb_2025` (de `targets_municipio`), `meta_t = meta_2026`;
   `predict_proba` → `reports/ranking_risco_2026.csv`
   (`co_municipio, no_municipio, sg_uf, p_nao_atingir_2026, meta_2026, pc_2025, esforco_2026`),
   ordenado desc. Top-50 vai para o README.
7. Sanidade: `Spearman(p_nao_atingir_2026, esforco_2026) > 0` forte; quem não
   atingiu 2025 deve ter risco médio 2026 maior. Registrar em D13.
8. Salvar `models/modelo_risco_meta.joblib`; `reports/resultados_modelo_risco_meta.md`;
   imagens `modelo_meta_01..`; `PROGRESS.md`; PR
   `feat(model): modelo de risco de não atingimento de meta com backtest 2025 e projeção 2026`.

Cuidados já conhecidos: `atingiu_meta_2025` é boolean nullable → `dropna().astype(int)`;
`porte` é `category` com NaN; `groupby(observed=True)`.

### Fluxo de PR usado (repetir)
```powershell
git checkout -b feature/<nome> -q
# ... trabalho ...
git add -A; git commit -q -m "<mensagem>"
git push -u origin feature/<nome>
gh pr create --title "<titulo>" --body "<corpo>" --base main
gh pr merge --squash --delete-branch
git checkout main -q; git pull -q
```

---

## 6. Lembretes para as fases seguintes

- **Fase 4 (aluno) — feito.** Não retreinar sem motivo (33 min). Se precisar do
  modelo: `joblib.load(MODELS_DIR / "modelo_aluno.joblib")["pipeline"]`.
- **Fase 5 (meta):** renomear features para o esquema genérico em `t`
  (`resultado_t1`, `resultado_t2`, `delta_t1`, `meta_t`, `esforco_t`, `atingiu_t1`).
  Treino t=2025 → projeção t=2026 com `meta_2026` e `pc_alfabetizado_aeeb_2025`
  (que vem de `targets_municipio.parquet`). Gerar `reports/ranking_risco_2026.csv`.
  Sanidade: Spearman(`p_nao_atingir_2026`, `esforco_2026`) > 0.
- **Fase 6 (clusters):** excluir metas e UF/região do fit; k ∈ [3..8] por silhouette;
  validar H5 *a posteriori*.
- **Fase 7:** permutation importance + SHAP nos dois modelos (Modelo A: usar amostra
  ≤20k do holdout — SHAP em HistGB via `shap.TreeExplainer` funciona; passar `X` já
  transformado por `pipe.named_steps["pre"]` e nomes via `get_feature_names`). Marcar
  H1–H5 como confirmada/refutada/parcial em `reports/resultados_modelos.md`.
- **Fase 8 (opcional):** recalibrar Modelo A (`CalibratedClassifierCV`, isotônica) se
  o README quiser citar probabilidades absolutas — hoje ECE 0,147.
- **Fase 8:** README com as 5 perguntas de negócio respondidas **com números** +
  `run_all.py` fim a fim + `docs/roteiro_video.md`.
