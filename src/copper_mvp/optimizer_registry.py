"""Registered G2b methods and a shared comparison request."""

from __future__ import annotations

from typing import Literal

import pymoo
from pydantic import BaseModel, ConfigDict, Field, model_validator

from copper_mvp.bayesian_optimizers import BAYESIAN_OPTIMIZERS, ENTROPY_OPTIMIZERS, bayesian_spec
from copper_mvp.contracts import RunRequest
from copper_mvp.optimizer_methods import EXTENDED_OPTIMIZERS, optimizer_method_spec
from copper_mvp.platypus_methods import PLATYPUS_OPTIMIZERS, platypus_method_spec
from copper_mvp.scalarization import SCALAR_OPTIMIZERS, scalarization_spec

LEGACY_OPTIMIZERS = ("NSGA-II", "SPEA2", "SMS-EMOA")
OPTIMIZERS = LEGACY_OPTIMIZERS + EXTENDED_OPTIMIZERS + PLATYPUS_OPTIMIZERS + SCALAR_OPTIMIZERS + BAYESIAN_OPTIMIZERS


def optimizer_catalog():
    descriptions = (
        ("NSGA-II", "non-dominated sorting and crowding distance", 64),
        ("SPEA2", "strength fitness, density and archive truncation", 64),
        ("SMS-EMOA", "fixed-reference hypervolume contribution survival", 1),
    )
    return {
        "schema_version": "optimizer-registry.g6f.v1",
        "new_optimizer_count": len(OPTIMIZERS) - 1,
        "items": [
            {
                "optimizer_id": name,
                "mechanism": mechanism,
                "package": "pymoo",
                "package_version": pymoo.__version__,
                "population": 64,
                "offspring_batch": offspring,
                "variables": "continuous",
                "objectives": 2,
                "inequality_constraints": True,
                "status": "registered",
                "execution_authorized": False,
            }
            for name, mechanism, offspring in descriptions
        ]
        + [
            {
                **optimizer_method_spec(name),
                "population": 64,
                "offspring_batch": 1 if name == "MOEA-D" else 64,
                "variables": "continuous",
                "objectives": 2,
                "inequality_constraints": True,
                "package": "pymoo",
                "package_version": pymoo.__version__,
                "status": "registered",
                "execution_authorized": False,
            }
            for name in EXTENDED_OPTIMIZERS
        ]
        + [
            {
                **platypus_method_spec(name),
                "population": 1 if name == "PAES" else 64,
                "offspring_batch": 1 if name == "PAES" else 2 if name == "Epsilon-MOEA" else 64,
            }
            for name in PLATYPUS_OPTIMIZERS
        ]
        + [scalarization_spec(name) for name in SCALAR_OPTIMIZERS]
        + [bayesian_spec(name) for name in BAYESIAN_OPTIMIZERS],
    }


class OptimizerComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    mode: Literal["plant", "benchmark"] = "plant"
    benchmark_problem: Literal["constrained_quadratic", "unconstrained_quadratic"] = "constrained_quadratic"
    event_ids: tuple[str, ...] = ()
    optimizers: tuple[str, ...] = LEGACY_OPTIMIZERS
    seeds: tuple[int, ...] = (20260911, 20260912, 20260913)
    model_profile: Literal["DeltaHGB", "DeltaRidge"] = "DeltaHGB"
    model_scope: Literal["oof_replay", "development_analysis"] = "oof_replay"
    bundle_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    radius: float = Field(default=0.1, ge=0.01, le=0.2)
    epsilon_as: float = Field(default=0, ge=0, le=5000)
    total_budget: int = Field(default=2048, ge=128, le=16384)
    seconds_per_run: int = Field(default=120, ge=5, le=600)

    @model_validator(mode="after")
    def scope(self):
        if (
            not self.optimizers
            or len(set(self.optimizers)) != len(self.optimizers)
            or any(x not in OPTIMIZERS for x in self.optimizers)
        ):
            raise ValueError("优化器必须唯一且已注册")
        if (
            not 1 <= len(self.seeds) <= 10
            or len(set(self.seeds)) != len(self.seeds)
            or any(s < 0 or s > 2**31 - 1 for s in self.seeds)
        ):
            raise ValueError("种子列表必须唯一且有效，最多十个")
        if len(self.event_ids) > 20 or len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("工况列表必须唯一，最多二十个")
        if self.mode == "plant" and self.benchmark_problem != "constrained_quadratic":
            raise ValueError("工厂比较不接受数学问题选择")
        if any(name in ENTROPY_OPTIMIZERS for name in self.optimizers) and (
            self.mode != "benchmark" or self.benchmark_problem != "unconstrained_quadratic"
        ):
            raise ValueError("MES/JES 首版必须显式选择无约束数学问题")
        if self.mode == "plant" and "NBI" in self.optimizers and self.model_profile != "DeltaRidge":
            raise ValueError("NBI 当前仅在 DeltaRidge 历史代理上启用")
        if self.mode == "plant" and not self.event_ids:
            raise ValueError("请选择历史工况")
        if self.mode == "benchmark" and self.event_ids:
            raise ValueError("数学比较不接受工厂事件")
        return self

    def problem_request(self, event_id, seed):
        return RunRequest(
            task_type="optimize",
            request_key=self.request_key,
            mode=self.mode,
            event_id=event_id,
            model_profile=self.model_profile,
            model_scope=self.model_scope,
            bundle_id=self.bundle_id,
            radius=self.radius,
            epsilon_as=self.epsilon_as,
            seed=seed,
            evaluation_budget=64,
        )
