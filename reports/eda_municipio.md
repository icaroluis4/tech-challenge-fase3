# EDA Municipal — Resultados e Hipóteses

> Fontes: `notebooks/01_eda_municipio.ipynb` e `02_eda_aluno.ipynb` (executados, figuras em `images/`).

## Achados principais

1. **Desigualdade regional forte.** Média municipal de alfabetizados 2025:
   S 72,8% · SE 73,5% · CO 81,2% · NE 69,6% · N 62,5%. O Norte é a região mais
   atrasada; o CO avançou ~16,7 p.p. entre 2023 e 2025 (maior salto).
2. **Inércia do resultado.** A correlação de Spearman mais forte com o
   resultado 2025 é o próprio resultado 2024 — o passado recente domina.
3. **Atingimento de meta 2025: 72,5%** dos municípios (3.927 de 5.417
   rotulados). Municípios que não atingiram tinham **esforço exigido maior**
   (meta − resultado 2024), confirmando que metas ambiciosas demais falham mais.
4. **Contexto socioeconômico importa.** IDHM-Educacional, renda per capita e
   taxa de analfabetismo adulto aparecem entre as maiores correlações
   absolutas com o resultado (ver `eda_04_correlacao_spearman.png`).
5. **ICC municipal ≈ 0,08** (notebook 02): apenas ~8% da variância do target
   individual está *entre* municípios; ~92% é intra-município (escola,
   família, indivíduo) — não observada. Isso define o **teto realista** do
   Modelo A e justifica enquadrá-lo como modelo de risco contextual.
6. **Leakage confirmado visualmente** (`eda_aluno_02_leakage_proficiencia.png`):
   separação perfeita em 743 — nenhum não-alfabetizado acima, nenhum
   alfabetizado abaixo.

## Hipóteses analíticas (→ veredito na Fase 7)

| # | Hipótese | Implicação de modelagem |
|---|----------|-------------------------|
| H1 | Resultado 2024 é o preditor dominante (inércia) | Feature `pc_alfabetizado_2024` deve liderar importância; ablação sem histórico quantifica o valor do contexto |
| H2 | IDHM-E e renda explicam parte residual do resultado | Features socioeconômicas devem aparecer no top-10 de importância |
| H3 | Ruralidade reduz a probabilidade de alfabetização | `pct_escolas_rurais` com efeito negativo no SHAP |
| H4 | Metas mais ambiciosas (esforço alto) reduzem atingimento | `esforco_t` deve ser preditor forte do Modelo B |
| H5 | N/NE concentram risco educacional | Ranking 2026 e clusters devem concentrar municípios N/NE no grupo vulnerável |

## Decisões derivadas

- Modelo A reporta **GroupKFold por município** além do split aleatório — com
  ICC baixo, o número honesto para o gestor é o de generalização territorial.
- Modelo B inclui `esforco_t` como feature central (H4 já tem evidência na EDA).
- Clusters excluem UF/região do fit para não forçar geografia; a concentração
  regional (H5) é validada *a posteriori* como validação externa.
