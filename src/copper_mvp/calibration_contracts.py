"""Fixed time-split calibration protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copper_mvp.model_registry import method_spec

DEFAULT_CALIBRATION_METHODS = [
    "Persistence",
    "DeltaRidge",
    "BayesianRidge",
    "MultiTaskElasticNet",
    "Quantile",
    "NGBoost",
]


class CalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    methods: list[str] = Field(default_factory=lambda: list(DEFAULT_CALIBRATION_METHODS), min_length=1, max_length=38)
    seeds: list[int] = Field(
        default_factory=lambda: [20260911, 20260912, 20260913, 20260914, 20260915], min_length=1, max_length=10
    )
    coverage: Literal[0.9] = 0.9
    calibration_fraction: float = Field(default=0.2, ge=0.1, le=0.4)
    minimum_calibration: int = Field(default=60, ge=30, le=300)
    minimum_fit: int = Field(default=120, ge=60, le=1000)
    max_wall_seconds: int = Field(default=7200, ge=60, le=86400)

    @model_validator(mode="after")
    def registered(self):
        if len(set(self.methods)) != len(self.methods) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("方法和种子不能重复")
        if "Persistence" not in self.methods:
            raise ValueError("比较必须包含 Persistence 参考")
        for method in self.methods:
            method_spec(method, self.seeds[0])
        return self
