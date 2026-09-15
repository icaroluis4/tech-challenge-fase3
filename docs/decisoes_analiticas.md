# Decisões Analíticas — Tech Challenge Fase 3

> Padrão herdado da Fase 2: toda decisão com impacto analítico é registrada aqui.

## D1 — Modelo de aluno treina apenas com presentes (`IN_PRESENCA_LP = 1`)

**Contexto:** 252.871 alunos ausentes recebem `IN_ALFABETIZADO = 0` por regra
administrativa (não por avaliação). Incluí-los misturaria dois processos
geradores distintos (ausência × proficiência real) e inflaria artificialmente
a performance do modelo (ausente → sempre classe 0).

**Decisão:** treino/validação apenas com os 1.969.921 presentes (66,2%
alfabetizados). Ausência é tratada como problema separado (desenho de
política de participação), fora do escopo supervisionado.

## D2 — ADH/IDHM de 2010 como proxy estrutural

**Contexto:** o Atlas do Desenvolvimento Humano municipal mais recente
disponível na Base dos Dados é o censo 2010.

**Decisão:** usar `idhm`, `idhm_e`, `idhm_r`, `indice_gini`, `renda_pc`,
`taxa_analfabetismo_15_mais` como proxies **estruturais** (lentamente
variantes), não conjunturais. Limitação registrada no README. Complementado
com PIB 2023 e população 2024 (conjuntura recente).

## D3 — Sem features de escola no modelo de aluno

**Contexto:** `ID_ESCOLA` na Bronze AEEB é anonimizado — 0% de join com
`censo_escola.co_entidade` (verificado na Fase 2, fato F4).

**Decisão:** o aluno recebe apenas contexto do **município** (feature store
municipal) + `tp_dependencia` (rede) + UF. Consequência: o modelo é de
**risco contextual**, não classificador individual — o teto de AUC esperado
(0,62–0,72) reflete a variância intra-município não observada (família,
escola). Isso é explicitado no README e sustentado pelo cálculo de ICC na EDA.

## D4 — Princípio anti-leakage único

**Contexto:** F1 (proficiência ≥ 743 ⇔ alfabetizado) e F6 (agregados 2025 da
Gold derivam do resultado que se quer prever).

**Decisão:** features só contêm informação disponível **antes** da avaliação
2025 — resultados 2023/2024, metas (definidas a priori), Censo Escolar 2024,
IBGE, ADH. Listas negras hard-coded em `src/preprocessing/leakage.py` com
teste unitário que falha se qualquer coluna proibida entrar em `X`.
Evidência empírica do corte 743 em `reports/evidencia_leakage_proficiencia.md`.

## D5 — Targets 2025 separados das features

**Decisão:** `targets_municipio.parquet` guarda `atingiu_meta_2025`,
`pc_alfabetizado_aeeb_2025`, `gap_meta_2025` em dataframe à parte, joinável
por `co_municipio` apenas na etapa de treino/avaliação — nunca na feature
store. Isso torna o leakage estruturalmente impossível no pipeline.

## D6 — Modelo B formulado genericamente em t (backtest + projeção)

**Contexto:** não existe resultado 2026 (F10). "Prever metas futuras" exige
treinar com o passado e aplicar deslocando as features.

**Decisão:** features nomeadas genericamente (`resultado_t1`, `meta_t`,
`esforco_t`…). Treino: t=2025 (resultado_t1=pc_2024, target=atingiu_meta_2025).
Projeção: t=2026 (resultado_t1=pc_2025, meta_t=meta_2026). O mesmo pipeline
serializado serve aos dois momentos.

## D7 — Correções na feature store descobertas pelos testes de pipeline (Fase 3)

**Contexto:** ao ajustar o `ColumnTransformer` em dados reais, o `SimpleImputer`
avisou que `va`, `va_agropecuaria`, `va_industria`, `va_servicos` e
`share_va_agro` não tinham nenhum valor observado. Investigação na fonte
(`basedosdados.br_ibge_pib.municipio`): o valor adicionado setorial só é
publicado até **2021**; em 2022/2023 só existe `pib`. Além disso,
`pib_per_capita_2023` estava com mediana ≈ R$ 29 milhões — o PIB da fonte
já está em R$ correntes, não em mil R$.

**Decisão:**
1. `pib` continua de 2023 (último ano); VA setorial passa a vir de **2021**
   (defasagem de 2 anos, proxy estrutural da composição econômica).
2. `pib_per_capita_2023 = pib / populacao_2024` (sem `× 1000`). Mediana
   resultante ≈ R$ 28,9 mil, coerente com IBGE.
3. `share_va_agro` pode ser negativo (VA agropecuário negativo em ~1% dos
   municípios); o ramo `log` do pré-processador usa `symlog1p` para não gerar NaN.
4. Novos testes em `tests/test_features.py`: nenhuma coluna 100% nula e
   PIB per capita dentro de escala plausível.

## D8 — Pré-processamento dentro do `Pipeline` e duas visões de validação

**Decisão:** `build_preprocessor()` (imputação mediana/moda + `symlog1p` em
contagens + `StandardScaler` + `OneHotEncoder(handle_unknown="ignore")`) é
sempre o primeiro step do `Pipeline` do estimador — imputação e scaling só
aprendem no `fit` do treino e são serializados junto com o modelo (`joblib`).
Cada modelo reporta **holdout estratificado 80/20 + `StratifiedKFold(5)`**
(desempenho médio) **e `GroupKFold`** por município (aluno) ou UF (município)
— generalização para território nunca visto, que é o número relevante para
o gestor público.

