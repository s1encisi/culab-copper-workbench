"""Versioned method catalogue and fixed G2a comparison protocol."""
from __future__ import annotations

from typing import Literal
import sklearn
from pydantic import BaseModel, ConfigDict, Field, model_validator

from copper_mvp.common import WorkbenchError, digest
from copper_mvp.classical_registry import CLASSICAL_METHODS, classical_spec

REGISTRY_VERSION = "model-registry.g2a.v1"
LEGACY_METHOD_IDS = ("Persistence", "DeltaRidge", "DeltaHGB", "ElasticNet", "Huber", "PLS")
METHOD_IDS = LEGACY_METHOD_IDS + CLASSICAL_METHODS
FEATURE_COUNT = 114
NUMERIC_COUNT = 110
SEED = 20260905
PRESETS = {
    "Persistence": {},
    "DeltaRidge": {"alpha": 1000.0},
    "DeltaHGB": {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15,
                 "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False},
    "ElasticNet": {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 20000, "tol": 1e-6, "selection": "cyclic"},
    "Huber": {"epsilon": 1.35, "alpha": 0.0001, "max_iter": 10000, "tol": 1e-5},
    "PLS": {"n_components": 5, "scale": False, "max_iter": 500, "tol": 1e-6},
}
IMPLEMENTATIONS = {
    "Persistence": "current_result_reference",
    "DeltaRidge": "sklearn.linear_model.Ridge",
    "DeltaHGB": "sklearn.ensemble.HistGradientBoostingRegressor",
    "ElasticNet": "sklearn.linear_model.ElasticNet",
    "Huber": "sklearn.linear_model.HuberRegressor",
    "PLS": "sklearn.cross_decomposition.PLSRegression",
}


def method_spec(method_id: str, seed: int = SEED) -> dict:
    if method_id in CLASSICAL_METHODS:
        return classical_spec(method_id, seed)
    if method_id not in METHOD_IDS:
        raise WorkbenchError("没有该注册方法", "METHOD_NOT_FOUND")
    spec = {
        "schema_version": REGISTRY_VERSION, "method_id": method_id,
        "method_version": "g2a.fixed.v2" if method_id == "Huber" else "g2a.fixed.v1", "implementation": IMPLEMENTATIONS[method_id],
        "package_version": sklearn.__version__, "requires_fit": method_id != "Persistence",
        "status": "registered", "feature_count": FEATURE_COUNT,
        "targets": [{"name": "cu", "unit": "g/L"}, {"name": "as", "unit": "mg/L"}],
        "target_transform": "current_result" if method_id == "Persistence" else "delta_from_current",
        "target_scaling": "train_only_standard_delta" if method_id in ("ElasticNet", "Huber", "PLS") else "legacy_unchanged",
        "multi_output": "joint" if method_id == "PLS" else "reference" if method_id == "Persistence" else "two_independent_estimators",
        "preprocessing": "none" if method_id == "Persistence" else "native_numeric_missing_and_categorical" if method_id == "DeltaHGB"
                         else "train_median_keep_empty_standardize_numeric_fixed_one_hot_modes",
        "preset_parameters": dict(PRESETS[method_id]), "seed": seed,
        "forecast_use": "research_comparison", "optimization_proxy_approval": "not_granted_by_g2a",
        "causal_control": False, "automatic_promotion": False,
    }
    spec["content_hash"] = digest(spec)
    return spec


def catalog() -> dict:
    return {"schema_version": "model-registry.g6a.v1", "items": [method_spec(m) for m in METHOD_IDS],
            "registered_count": len(METHOD_IDS), "new_method_count": 3 + len(CLASSICAL_METHODS), "automatic_promotion": False,
            "default_comparison_methods": list(LEGACY_METHOD_IDS)}


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    methods: tuple[str, ...] = LEGACY_METHOD_IDS
    seed: int = Field(default=SEED, ge=0, le=2**31 - 1)
    max_wall_seconds: int = Field(default=900, ge=30, le=1800)

    @model_validator(mode="after")
    def validate_methods(self):
        if (len(set(self.methods)) != len(self.methods) or not self.methods
            or any(m not in METHOD_IDS for m in self.methods) or "Persistence" not in self.methods):
            raise ValueError("方法必须唯一、已注册，且包含 Persistence 参照")
        return self


class ComparisonPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: str = Field(min_length=1, max_length=180)
    method_id: str = Field(min_length=1, max_length=80)
    scope: Literal["oof_replay", "development_analysis"] = "oof_replay"
    include_uncertainty: bool = False

    @model_validator(mode="after")
    def registered_method(self):
        if self.method_id not in METHOD_IDS:
            raise ValueError("请选择已注册方法")
        return self
