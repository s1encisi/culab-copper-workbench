"""正式工程的可移植路径定义。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_DATA_DIR = PROJECT_ROOT / "data" / "development_2024_2025"
FROZEN_CONTRACT_DIR = PROJECT_ROOT / "contracts" / "frozen"
P2_RUN_MANIFEST = DEVELOPMENT_DATA_DIR / "p2_run_manifest_v1.json"
SELECTED_MODEL_MANIFEST = PROJECT_ROOT / "models" / "selected_model_manifest_v1.json"
RUNS_DIR = PROJECT_ROOT / "runs"


__all__ = [
    "DEVELOPMENT_DATA_DIR",
    "FROZEN_CONTRACT_DIR",
    "P2_RUN_MANIFEST",
    "PROJECT_ROOT",
    "RUNS_DIR",
    "SELECTED_MODEL_MANIFEST",
]
