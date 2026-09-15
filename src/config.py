"""Configuração central do projeto (paths, GCP, seed)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- GCP / BigQuery ---
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "semiotic-primer-366516")
BQ_DATASET_GOLD: str = os.getenv("BQ_DATASET_GOLD", "gold_alfabetizacao")
BQ_DATASET_BRONZE: str = os.getenv("BQ_DATASET_BRONZE", "bronze_alfabetizacao")
BQ_DATASET_ML: str = os.getenv("BQ_DATASET_ML", "gold_ml")

TBL_GOLD_MUNICIPIO = f"{GCP_PROJECT_ID}.{BQ_DATASET_GOLD}.indicador_municipio"
TBL_GOLD_UF = f"{GCP_PROJECT_ID}.{BQ_DATASET_GOLD}.indicador_uf"
TBL_ALUNO = f"{GCP_PROJECT_ID}.{BQ_DATASET_BRONZE}.aeeb_ts_aluno"

# --- Paths ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "images"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reprodutibilidade ---
SEED: int = int(os.getenv("SEED", "42"))
