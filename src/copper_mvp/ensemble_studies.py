"""Ensemble studies reuse the established private research-job lifecycle."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator

from copper_mvp.common import WorkbenchError, digest, file_hash, write_json
from copper_mvp.ensemble_experiment import ENSEMBLE_FILES, run_ensemble_experiment
from copper_mvp.ensemble_models import EnsembleSettings
from copper_mvp.routing_studies import RoutingRequest, RoutingStudies


class EnsembleRequest(RoutingRequest):
    max_wall_seconds: int = Field(default=3600, ge=60, le=7200)
    seeds: tuple[int, ...] = (20260911, 20260912, 20260913, 20260914, 20260915)

    @model_validator(mode="after")
    def seed_scope(self):
        if (
            not 1 <= len(self.seeds) <= 8
            or len(set(self.seeds)) != len(self.seeds)
            or any(s < 0 or s >= 2**31 for s in self.seeds)
        ):
            raise ValueError("使用一到八个不重复的非负种子")
        return self


class EnsembleStudies(RoutingStudies):
    directory_name = "ensemble_studies"
    signature_files = ENSEMBLE_FILES

    def policy_spec(self):
        return EnsembleSettings().model_dump()

    def perform_study(self, root, request, progress):
        return run_ensemble_experiment(
            self.data, self.comparisons.directory(request.comparison_id), root, request, progress
        )

    def resume(self, actor, identifier, executor=None):
        actor.require("compute")
        with self.lock:
            state = self.get(actor, identifier, result=False)
            if state["status"] not in ("failed", "interrupted"):
                raise WorkbenchError("只恢复已失败或中断的研究", "ENSEMBLE_STATE")
            request = EnsembleRequest.model_validate(state["request"])
            source = self.comparisons.get(request.comparison_id, result=False)
            expected = digest(
                {
                    "request": request.model_dump(),
                    "source": source["fingerprint"],
                    "policy": self.policy_spec(),
                    "code": {name: file_hash(Path(__file__).with_name(name)) for name in self.signature_files},
                }
            )
            if state["fingerprint"] != expected:
                raise WorkbenchError("来源或训练实现已变化，请使用新请求键", "SOURCE_CHANGED")
            state.update(status="queued", owner_pid=os.getpid())
            state.pop("error", None)
            write_json(self.directory(identifier) / "state.json", state)
        if executor is None:
            self.execute(identifier, request)
            return self.get(actor, identifier)
        executor.submit(self.execute, identifier, request)
        return state
