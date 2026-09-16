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

## D9 — Threshold por custo assimétrico 5:1 (falso negativo de risco pesa mais)

**Contexto:** o threshold 0,5 é arbitrário e, com 66,2% de positivos, joga o
modelo para "prever alfabetizado". Os dois erros têm consequências muito
diferentes em política pública:

| Erro | Significado | Custo real |
|------|-------------|------------|
| `miss_risco` (y=0 previsto como 1) | criança em risco **não** é sinalizada | perde-se a janela de intervenção no ciclo de alfabetização; dano é cumulativo e de difícil reversão |
| `falso_alarme` (y=1 previsto como 0) | criança já alfabetizada recebe reforço | desperdício marginal de recurso pedagógico, sem dano à criança |

**Decisão:** além do threshold que maximiza F1, reportar e adotar como
threshold operacional o que **minimiza o custo esperado** com razão
**5:1** (`CUSTO_MISS_RISCO=5`, `CUSTO_FALSO_ALARME=1`) em
`best_threshold_cost()`. A razão 5:1 é uma escolha de política — declarada,
não estimada — e é parâmetro do código: o gestor pode recalibrá-la conforme
o orçamento de reforço disponível. Ambos os thresholds são serializados no
`joblib` junto com o pipeline.

**Consequência:** o threshold ótimo por custo fica **acima** de 0,5
(≈0,70), o que aumenta `recall_risco` — captura-se mais crianças em risco ao
preço de mais falsos alarmes. Isso é o trade-off desejado.

## D10 — Métricas espelhadas na classe de risco

**Contexto:** a classe positiva do target (`in_alfabetizado = 1`) é a classe
majoritária e **não** é a classe de interesse. ROC-AUC e PR-AUC padrão
descrevem a habilidade de identificar quem *vai* se alfabetizar.

**Decisão:** `evaluate_binary()` reporta, além das métricas convencionais,
`pr_auc_risco`, `f1_risco`, `precision_risco` e `recall_risco`, calculadas
com `(1-y, 1-p)` — isto é, tratando **não alfabetizado como positivo**. As
tabelas do relatório mostram as duas direções; a leitura de política pública
usa a versão `_risco`. Regularização e calibração (`brier`, ECE) também são
reportadas porque a probabilidade será usada como **ranking de prioridade**,
não só como rótulo.

## D11 — Correlação municipal reportada com e sem corte de massa amostral

**Contexto:** a validação mais informativa do Modelo A é comparar a média das
probabilidades por município com a taxa observada de alfabetização. Mas a taxa
observada em uma amostra de `n` alunos tem desvio binomial ≈ 0,5/√n: com ~11
alunos por município (amostra de 400k em 5,4 mil municípios), o "observado" é
majoritariamente ruído, o que deprime a correlação **por construção**.

**Decisão:** reportar a correlação de Pearson/Spearman no conjunto completo
**e** restrita a municípios com `n_alunos ≥ 30` no holdout
(`_correlacoes_municipais`). A segunda é a leitura honesta da capacidade de
ordenar municípios por risco; a primeira é mantida para não dar impressão de
*cherry-picking*. O CSV completo
(`reports/modelo_aluno_agregado_municipal.csv`) permite auditoria.

## D12 — Amostragem estratificada de 400k para experimentação

**Contexto:** 1,94M × 55 features × 5 folds × 25 iterações de busca é
proibitivo em notebook local, e o ganho estatístico é nulo — o erro padrão de
uma AUC com 400k observações já é da ordem de 0,001.

**Decisão:** `sample_aluno(n=400_000)` estratificado por `(sg_uf,
in_alfabetizado)` preserva composição territorial e balanceamento de classes;
a comparação de candidatos e a busca de hiperparâmetros rodam em 150k
(`cv_sample`) e o **refit final** usa os 320k do treino do holdout. A base
completa pode ser usada com `--n-sample 0`. Toda a amostragem é semeada
(`SEED=42`) e coberta por teste de reprodutibilidade.



## D13 — Modelo B: ablação do bloco de histórico e leitura da projeção 2026

**Contexto:** prever `atingiu_meta` é "fácil demais" por inércia — quem já
está longe da meta tende a continuar longe. Sem ablação, não sabemos se o
modelo aprendeu algo além de `resultado_t1`. E a projeção 2026 precisa de
checagens de coerência antes de virar ranking de gestão.

**Decisão:**

1. **Ablação obrigatória** (`_ablacao_sem_historico`): o melhor modelo é
   retreinado sem `resultado_t1/t2`, `delta_t1`, `meta_t`, `esforco_t` e
   `atingiu_t1` — só Censo, IBGE, ADH e território. Resultado (holdout):
   AUC 0,852 → 0,790 sem histórico. Ou seja, boa parte do ganho sobre o acaso
   vem da inércia do resultado, mas o contexto socioeconômico **sozinho**
   ainda entrega AUC 0,79 — sinal preditivo real, não só repetição do passado.
2. **Sanidade da projeção 2026** (`predict_2026._sanidade`): Spearman entre
   `p_nao_atingir_2026` e `esforco_2026` = 0,78 (forte e positiva, como
   esperado); risco médio de quem falhou 2025 = 0,67 vs 0,14 de quem atingiu.
3. **Calibração importa aqui**: as probabilidades viram ranking de prioridade
   de gestão. ECE do modelo final = 0,018 (logreg sem `class_weight`) — bom o
   suficiente para ler as probabilidades como frequências aproximadas, sem
   necessidade de `CalibratedClassifierCV`.
4. **Concentração no RS é sinal, não artefato**: o RS teve a pior taxa de
   atingimento 2025 do país (27,9% vs 72,5% nacional) e esforço 2026 médio de
   +10,5 p.p. (Brasil: −3,4). O top-100 do ranking ter 88 municípios do RS
   reflete o choque real de 2025 no estado — registrado para defesa no vídeo.

## D14 — Escopo reduzido de entrega (decisão de última hora, urgente)

**Contexto:** na reta final do prazo restava ~1 hora de trabalho efetivo e
faltavam as Fases 6 (clusters), 7 (interpretabilidade) e 8 (README +
`run_all.py`). O plano original previa SHAP completo (beeswarm, dependence,
waterfall), UMAP/mapas de clusters, notebook de interpretabilidade e roteiro
de vídeo — inviáveis no tempo restante sem sacrificar a qualidade do núcleo.

**Decisão (acordada com o stakeholder):** entregar o **mínimo funcional que
cobre todos os requisitos nominais do enunciado**, nesta ordem de prioridade:

1. **Fase 6 enxuta** — KMeans k∈[3..8] por silhouette, sem UMAP/mapa; perfil
   dos clusters em tabela + `reports/clusters_municipios.csv`.
2. **Fase 7 light** — **permutation importance** nos dois modelos (satisfaz
   "interpretabilidade" e responde às perguntas de fatores/variáveis) +
   **SHAP apenas no Modelo B** (logreg, 5,4k linhas — `LinearExplainer` é
   instantâneo, cobre o requisito nominal de SHAP). SHAP do Modelo A
   (HistGB, 400k linhas) fica como evolução futura.
3. **Fase 8** — README respondendo às 5 perguntas de negócio **com números já
   apurados** + `run_all.py` reproduzível. `train_aluno` (33 min) fica fora
   do `run_all` padrão (flag opcional), documentado.
4. **Cortes explícitos:** mapas/UMAP, notebook de interpretabilidade,
   recalibração do Modelo A (ECE 0,147 — documentado como limitação),
   roteiro de vídeo e apresentação.

**Justificativa:** todos os números necessários às 5 perguntas já existem
(Fases 4–5); o que faltava era cobertura nominal dos requisitos
"clusterização", "SHAP" e "pipeline reproduzível". Preferiu-se cobrir todos
os requisitos com profundidade reduzida a entregar poucos requisitos com
profundidade total — o critério de avaliação é de conformidade + clareza,
não de exaustão técnica.
