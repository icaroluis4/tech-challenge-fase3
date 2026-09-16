# Clusterização de municípios — Fase 6

Fit sem metas, sem resultados 2025 e sem UF/região (D14: escopo reduzido,sem UMAP/mapa). Pré-processamento idêntico ao dos modelos (mediana →symlog1p/scaler + OHE).

## Escolha de k (silhouette)

|      k |   silhouette |     inercia |
|-------:|-------------:|------------:|
| 3.0000 |       0.2188 | 135915.9940 |
| 4.0000 |       0.1760 | 125244.0732 |
| 5.0000 |       0.1584 | 115738.0184 |
| 6.0000 |       0.1564 | 109641.2363 |
| 7.0000 |       0.1356 | 104979.8104 |
| 8.0000 |       0.1219 | 101562.3570 |

**k escolhido: 3** (maior silhouette).

## Clusters nomeados

|   cluster | nome_cluster                   |
|----------:|:-------------------------------|
|         0 | alto desempenho urbano pequeno |
|         1 | alto desempenho urbano médio   |
|         2 | baixo desempenho rural pequeno |

## Perfil (medianas por cluster)

|   cluster |   pc_alfabetizado_2024 |   pc_alfabetizado_2023 |   populacao_2024 |   pct_escolas_rurais |   pib_per_capita_2023 |   idhm_e |   alunos_por_turma_ai |
|----------:|-----------------------:|-----------------------:|-----------------:|---------------------:|----------------------:|---------:|----------------------:|
|         0 |                  72.00 |                  70.00 |          5693.00 |                 0.17 |              38119.36 |     0.59 |                 18.00 |
|         1 |                  62.00 |                  59.00 |         53778.00 |                 0.17 |              46499.90 |     0.64 |                 21.11 |
|         2 |                  54.00 |                  49.00 |         13786.00 |                 0.71 |              14871.97 |     0.47 |                 20.35 |

## Validação externa — taxa de atingimento da meta 2025 por cluster

(não usada no fit; Brasil = 72,5%)

|   cluster |   atingiu_meta_2025_% |
|----------:|----------------------:|
|         0 |                  73.2 |
|         1 |                  61.9 |
|         2 |                  77.8 |

## Distribuição por região (contagem)

|   cluster |   CO |   N |   NE |   S |   SE |
|----------:|-----:|----:|-----:|----:|-----:|
|         0 |  311 |  95 |   68 | 850 |  944 |
|         1 |  125 |  56 |  187 | 295 |  512 |
|         2 |   31 | 299 | 1535 |  19 |  173 |
