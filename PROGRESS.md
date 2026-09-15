# PROGRESS — Tech Challenge Fase 3

> **Este arquivo é o ponto de retomada entre sessões.** Leia-o inteiro antes de
> continuar. O plano completo está em `../PLANO_IMPLEMENTACAO.md` (pasta `Desafio3/`).
> Atualize este arquivo ao fim de cada fase.

Última atualização: 2026-09-15 — **Fases 0, 1 e 2 concluídas. Próxima: Fase 3.**

---

## 1. Onde estamos

| Fase | Descrição | Status |
|------|-----------|--------|
| 0 | Setup (estrutura, venv 3.11, git, repo remoto) | ✅ concluída |
| 1 | Extração BQ + feature store municipal | ✅ concluída (PR #1) |
| 2 | EDA municipal e por aluno | ✅ concluída (PR #2) |
| **3** | **Pipeline de pré-processamento (`transformers.py`, `splits.py`) + testes** | ⏭️ **PRÓXIMA** |
| 4 | Modelo A — aluno | ⬜ |
| 5 | Modelo B — risco de meta + projeção 2026 | ⬜ |
| 6 | Clusterização de municípios | ⬜ |
| 7 | Interpretabilidade (permutation importance + SHAP) | ⬜ |
| 8 | README final, decisões, roteiro do vídeo, `run_all.py` | ⬜ |

Git: `main` em `844482c`. 2 PRs mergeados (squash). Repo:
`https://github.com/icaroluis4/tech-challenge-fase3` (privado).

---

## 2. Ambiente — comandos exatos

```powershell
cd "c:\Users\icaro\OneDrive\Área de Trabalho\Projetos\FIAP\Desafio3\tech-challenge-fase3"
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m src.data.build_features
.\.venv\Scripts\python.exe -m jupyter execute --inplace notebooks/01_eda_municipio.ipynb
```

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
| `src/visualization/plots.py` | tema seaborn + `save_fig(fig, path)` (backend Agg). |
| `tests/test_leakage.py` | 48 casos parametrizados. Verdes. |
| `tests/test_features.py` | grão único, 5.500 linhas, cobertura externos ≥98%, derivadas coerentes. Verdes. |

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

### Docs/relatórios
- `docs/decisoes_analiticas.md` — **D1..D6** já escritas.
- `reports/eda_municipio.md` — achados + hipóteses **H1–H5**.
- `reports/evidencia_leakage_proficiencia.md` — tabela min/max de proficiência por classe.
- `images/` — 11 arquivos: `eda_01..08` (município, um deles HTML do mapa) + `eda_aluno_01..03`.

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

## 5. Próximo passo — Fase 3 (detalhado)

Branch: `feature/preprocessing`.

1. **`src/preprocessing/transformers.py`**
   ```python
   def build_preprocessor(num_cols, cat_cols, scale=True): ...
   ```
   - numéricas: `SimpleImputer(median)` + `StandardScaler` (ou passthrough);
     `log1p` via `FunctionTransformer` em `populacao_2024`, `total_mat_fund_ai`, `qtd_escolas`.
   - categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore", min_frequency=20)`.
   - Categóricas do projeto: `sg_uf`, `regiao`, `porte`, `capital_uf`, `tp_dependencia` (aluno).
   - Helper para inferir `num_cols`/`cat_cols` a partir do DataFrame, já removendo
     `BLACKLIST` (usar `assert_no_leakage`).
2. **`src/preprocessing/splits.py`**
   - `split_aluno(df)`: holdout 80/20 estratificado por `in_alfabetizado`;
     `StratifiedKFold(5)` + **`GroupKFold(groups=co_municipio)`**.
   - `split_municipio(df)`: holdout 80/20 estratificado por `atingiu_meta_2025`
     (dropar nulos antes); `StratifiedKFold(5, shuffle=True, random_state=SEED)`;
     alternativa `GroupKFold(groups=sg_uf)`.
3. **`tests/test_pipeline.py`**: fit em ~1k linhas sintéticas com NaN →
   `predict_proba` sem erro; roundtrip `joblib.dump/load`.
4. Rodar `pytest tests -q` (devem continuar verdes) e abrir PR:
   `feat(preprocessing): ColumnTransformer + guard anti-leakage + estratégias de split`.

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

- **Fase 4 (aluno):** amostrar 400k estratificado por `(sg_uf, in_alfabetizado)`
  para experimentação; baselines Dummy → LogReg → HistGB/LGBM → RF;
  `RandomizedSearchCV(n_iter=25, scoring="roc_auc")`; reportar ROC-AUC, PR-AUC,
  F1, balanced acc, Brier, calibração, matriz de confusão **e GroupKFold**.
  AUC esperada 0,62–0,72 (ver G3) — explicar, não maquiar.
- **Fase 5 (meta):** renomear features para o esquema genérico em `t`
  (`resultado_t1`, `resultado_t2`, `delta_t1`, `meta_t`, `esforco_t`, `atingiu_t1`).
  Treino t=2025 → projeção t=2026 com `meta_2026` e `pc_alfabetizado_aeeb_2025`
  (que vem de `targets_municipio.parquet`). Gerar `reports/ranking_risco_2026.csv`.
  Sanidade: Spearman(`p_nao_atingir_2026`, `esforco_2026`) > 0.
- **Fase 6 (clusters):** excluir metas e UF/região do fit; k ∈ [3..8] por silhouette;
  validar H5 *a posteriori*.
- **Fase 7:** marcar H1–H5 como confirmada/refutada/parcial em `reports/resultados_modelos.md`.
- **Fase 8:** README com as 5 perguntas de negócio respondidas **com números** +
  `run_all.py` fim a fim + `docs/roteiro_video.md`.
