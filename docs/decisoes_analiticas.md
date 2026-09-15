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
