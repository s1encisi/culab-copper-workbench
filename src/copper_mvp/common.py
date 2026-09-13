from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_VERSION = "0.25.0"
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data/development_2024_2025"
EVIDENCE_DIR = WORKSPACE_ROOT / "04_多智能体项目/03_工程实现"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs/mvp"


class WorkbenchError(ValueError):
    def __init__(self, message: str, code: str = "INVALID_REQUEST"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LocalDataPaths:
    """Resolve private resources locally; HTTP clients cannot choose paths."""
    data_dir: Path
    evidence_dir: Path
    contract_dir: Path
    label_dir: Path

    @classmethod
    def resolve(cls, data_dir=None, evidence_dir=None, contract_dir=None, label_dir=None):
        def choose(explicit, variable, default):
            path = Path(explicit or os.environ.get(variable) or default).expanduser()
            return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
        data = choose(data_dir, "COPPER_MVP_DATA_DIR", DATA_DIR)
        evidence = choose(evidence_dir, "COPPER_MVP_EVIDENCE_DIR", EVIDENCE_DIR)
        contracts = choose(contract_dir, "COPPER_MVP_CONTRACT_DIR", PROJECT_ROOT / "configs/contracts")
        labels = choose(label_dir, "COPPER_MVP_LABEL_DIR", evidence / "artifacts/p1/development_2024_2025")
        return cls(data, evidence, contracts, labels)

    def sources(self) -> dict[str, Path]:
        return {
            "features": self.data_dir / "core_feature_matrix_v2.csv",
            "admissions": self.data_dir / "as_of_admission_card_core_v2.csv",
            "index": self.data_dir / "training_evaluation_index_v2.csv",
            "folds": self.data_dir / "cv_fold_manifest_v2.csv",
            "feature_contract": self.contract_dir / "core_features_v2.yaml",
            "timing_contract": self.contract_dir / "timing_contract_v2.yaml",
            "labels": self.label_dir / "outcome_ledger_v2.csv",
            "pairing": self.label_dir / "event_pair_index_v2.csv",
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value: Any) -> str:
    return json.dumps(safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(value) + "\n", encoding="utf-8")
    # Windows readers may briefly deny replacement while indexing a completed file.
    for delay in (0, 0.02, 0.05, 0.1, 0.2):
        if delay:
            time.sleep(delay)
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if os.name != "nt" or delay == 0.2:
                raise
