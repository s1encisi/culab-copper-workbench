from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: Literal["train", "predict", "optimize", "diagnostic"]
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    event_id: str | None = Field(default=None, max_length=180)
    model_scope: Literal["oof_replay", "development_analysis"] = "oof_replay"
    model_profile: Literal["auto", "Persistence", "DeltaRidge", "DeltaHGB"] = "auto"
    bundle_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    mode: Literal["plant", "benchmark"] = "plant"
    radius: float = Field(default=0.10, ge=0.01, le=0.20)
    epsilon_as: float = Field(default=0.0, ge=0.0, le=5000.0)
    evaluation_budget: int = Field(default=2048, ge=64, le=4096)
    seed: int = Field(default=20260905, ge=0, le=2**31 - 1)
    scenario: Literal["clean", "future_field", "model_mismatch", "invalid_numeric", "invalid_interval"] = "clean"

    @model_validator(mode="after")
    def needs_event(self):
        if (self.task_type == "predict" or (self.task_type == "optimize" and self.mode == "plant")) and not self.event_id:
            raise ValueError("请选择历史事件")
        return self


class ExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Literal["summary", "mode", "model", "tradeoff", "constraints"] = "summary"
    use_llm: bool = False


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1, max_length=80)
    label: str = Field(default="我的方案", min_length=1, max_length=60)


class AgentDiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Literal["why_few_candidates", "constraint_effect", "model_response"] = "why_few_candidates"
    request_key: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
