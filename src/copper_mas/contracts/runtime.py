"""多智能体运行计划与资源账本合同。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictRuntimeCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowPlanV1(_StrictRuntimeCard):
    schema_version: Literal["1.0"] = "1.0"
    graph_id: Literal["COPPER_MAS_A1_A5_V1"] = "COPPER_MAS_A1_A5_V1"
    run_id: str = Field(min_length=1)
    ordered_nodes: tuple[Literal["A1", "A2", "A4", "A5"], ...]
    numeric_predictor_ref: str = Field(min_length=1)
    external_test_access: Literal["DENIED", "POST_FREEZE_ONLY"]

    @model_validator(mode="after")
    def validate_frozen_order(self) -> WorkflowPlanV1:
        if self.ordered_nodes != ("A1", "A2", "A4", "A5"):
            raise ValueError("在线预测图节点顺序已冻结为 A1→A2→A4→A5")
        return self


class AgentExecutionRecordV1(_StrictRuntimeCard):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    agent_id: Literal["A1", "A2", "A3", "A4", "A5", "ORCHESTRATOR"]
    engine: Literal["DETERMINISTIC", "LOCAL_NUMERIC_MODEL", "KIMI", "DEEPSEEK"]
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0)
    status: Literal["PASSED", "WARNING", "FAILED", "ABSTAINED"]
    call_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_cny: float = Field(default=0.0, ge=0)
    cost_estimation_status: Literal[
        "NOT_APPLICABLE",
        "ESTIMATED_FROM_REPORTED_TOKENS",
        "CONSERVATIVE_UPPER_BOUND_NO_USAGE",
    ] = "NOT_APPLICABLE"
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_time_and_engine(self) -> AgentExecutionRecordV1:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at 不能早于 started_at")
        if self.engine in {"DETERMINISTIC", "LOCAL_NUMERIC_MODEL"}:
            if self.call_count or self.input_tokens or self.output_tokens or self.estimated_cost_cny:
                raise ValueError("本地确定性节点的外部调用资源必须为零")
            if self.cost_estimation_status != "NOT_APPLICABLE":
                raise ValueError("本地确定性节点不得声明外部 token 成本")
        return self


class ResourceLedgerV1(_StrictRuntimeCard):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    records: tuple[AgentExecutionRecordV1, ...]
    total_latency_ms: float = Field(ge=0)
    total_calls: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_estimated_cost_cny: float = Field(ge=0)
    run_cost_limit_cny: float | None = Field(default=None, gt=0)
    cost_limit_status: Literal["NOT_CONFIGURED", "WITHIN_LIMIT"] = "NOT_CONFIGURED"

    @model_validator(mode="after")
    def validate_cost_limit(self) -> ResourceLedgerV1:
        if self.run_cost_limit_cny is None:
            if self.cost_limit_status != "NOT_CONFIGURED":
                raise ValueError("未配置单次成本上限时状态必须为 NOT_CONFIGURED")
        else:
            if self.total_estimated_cost_cny > self.run_cost_limit_cny + 1e-12:
                raise ValueError("资源账本估算成本超过单次运行硬上限")
            if self.cost_limit_status != "WITHIN_LIMIT":
                raise ValueError("已配置成本上限时状态必须为 WITHIN_LIMIT")
        return self

    @classmethod
    def from_records(
        cls,
        run_id: str,
        records: tuple[AgentExecutionRecordV1, ...],
        *,
        run_cost_limit_cny: float | None = None,
    ) -> ResourceLedgerV1:
        return cls(
            run_id=run_id,
            records=records,
            total_latency_ms=sum(item.latency_ms for item in records),
            total_calls=sum(item.call_count for item in records),
            total_input_tokens=sum(item.input_tokens for item in records),
            total_output_tokens=sum(item.output_tokens for item in records),
            total_estimated_cost_cny=sum(item.estimated_cost_cny for item in records),
            run_cost_limit_cny=run_cost_limit_cny,
            cost_limit_status=("WITHIN_LIMIT" if run_cost_limit_cny is not None else "NOT_CONFIGURED"),
        )
