# Dados

Os dados não são versionados (ver `.gitignore`). Para regenerar:

```bash
python -m src.data.extract_bq        # BQ -> data/raw/*.parquet
python -m src.data.build_features    # data/raw -> data/processed/*.parquet
```

Pré-requisito: ADC do GCP configurado (`gcloud auth application-default login`)
e `.env` preenchido a partir de `.env-example`.

## Arquivos gerados

| Arquivo | Origem | Conteúdo |
|---------|--------|----------|
| `raw/gold_municipio.parquet` | `gold_alfabetizacao.indicador_municipio` | 5.500 municípios, 47 colunas |
| `raw/gold_uf.parquet` | `gold_alfabetizacao.indicador_uf` | 27 UFs |
| `raw/externos_municipio.parquet` | Base dos Dados (IBGE pop/PIB, ONU ADH, diretórios) | enriquecimento socioeconômico |
| `raw/aluno_presente.parquet` | `bronze_alfabetizacao.aeeb_ts_aluno` (presentes) | ~1,97M alunos |
| `processed/features_municipio.parquet` | Gold + externos, sem leakage 2025 | feature store municipal |
| `processed/targets_municipio.parquet` | Gold | targets 2025 (separados) |
| `processed/dataset_aluno.parquet` | aluno + features municipais | dataset do Modelo A |
