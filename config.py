from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
DATASET_VERSION = "v-2026-06-16_19-53-59"

ACTIVE_DATASET_DIR = DATASETS_DIR / DATASET_VERSION

