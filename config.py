from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent[0]

DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
DATASET_VERSION = "v-2026-02-07_15-53-36"

ACTIVE_DATASET_DIR = DATASETS_DIR / DATASET_VERSION
