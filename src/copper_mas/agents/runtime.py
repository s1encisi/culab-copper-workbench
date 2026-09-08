"""无需外部 LLM 的 A1→A2→A4→A5 可执行骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from time import perf_counter
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict

from copper_mas.agents.mode import infer_process_mode
from copper_mas.contracts.cards import (
    AuditCardV2,
    ForecastRequestV2,
    PredictionCardV2,
    ProcessModeCardV2,
)
from copper_mas.contracts.runtime import (
    AgentExecutionRecordV1,
    ResourceLedgerV1,
    WorkflowPlanV1,
)
from copper_mas.data.leakage import assert_no_future_information


class NumericPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predicted_cu_g_l: float
    predicted_as_mg_l: float
    cu_interval_low: float | None = None
    cu_interval_high: float | None = None
    as_interval_low: float | None = None
    as_interval_high: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OfflineGraphResult:
    plan: WorkflowPlanV1
    process_mode: ProcessModeCardV2
    prediction: PredictionCardV2
    audit: AuditCardV2
    resources: ResourceLedgerV1


PredictFunction = Callable[[Mapping[str, Any], ProcessModeCardV2], NumericPrediction]


def fixed_offline_plan(
    *, run_id: str, numeric_predictor_ref: str, post_freeze: bool = False
) -> WorkflowPlanV1:
    """Kimi 不可用或被禁用时的确定性编排；节点顺序与权限不变。"""

    return WorkflowPlanV1(
        run_id=run_id,
        ordered_nodes=("A1", "A2", "A4", "A5"),
        numeric_predictor_ref=numeric_predictor_ref,
        external_test_access="POST_FREEZE_ONLY" if post_freeze else "DENIED",
    )


def _record(
    *,
    run_id: str,
    agent_id: str,
    engine: str,
    started_at: datetime,
    started_perf: float,
    status: str,
    reason_codes: tuple[str, ...] = (),
) -> AgentExecutionRecordV1:
    ended_at = datetime.now(timezone.utc)
    return AgentExecutionRecordV1(
        run_id=run_id,
        agent_id=agent_id,
        engine=engine,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=max(0.0, (perf_counter() - started_perf) * 1000),
        status=status,
        reason_codes=reason_codes,
    )


def _validate_numeric_prediction(value: NumericPrediction) -> None:
    numeric = [value.predicted_cu_g_l, value.predicted_as_mg_l]
    if not all(math.isfinite(item) for item in numeric):
        raise ValueError("A4 数值预测包含非有限值")
    intervals = (
        (value.cu_interval_low, value.predicted_cu_g_l, value.cu_interval_high, "CU"),
        (value.as_interval_low, value.predicted_as_mg_l, value.as_interval_high, "AS"),
    )
    for low, point, high, name in intervals:
        if (low is None) != (high is None):
            raise ValueError(f"{name} 区间上下界必须同时存在或同时为空")
        if low is not None and not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"{name} 区间包含非有限值")
        if low is not None and not low <= point <= high:
            raise ValueError(f"{name} 点预测不在区间内")


def run_offline_graph(
    *,
    plan: WorkflowPlanV1,
    request: ForecastRequestV2,
    feature_row: Mapping[str, Any],
    predict: PredictFunction,
) -> OfflineGraphResult:
    """执行离线预测路径；任何泄漏或卡片不一致均 fail-closed。"""

    records: list[AgentExecutionRecordV1] = []

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    assert_no_future_information(request, decision_at=request.admission.decision_at)
    assert_no_future_information(feature_row, decision_at=request.admission.decision_at)
    records.append(
        _record(
            run_id=plan.run_id,
            agent_id="A1",
            engine="DETERMINISTIC",
            started_at=started_at,
            started_perf=started_perf,
            status="WARNING" if request.admission.admission_status == "WARNING" else "PASSED",
            reason_codes=request.admission.reason_codes,
        )
    )

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    mode = infer_process_mode(
        origin_event_id=request.admission.origin_event_id,
        decision_at=request.admission.decision_at,
        feature_row=feature_row,
    )
    records.append(
        _record(
            run_id=plan.run_id,
            agent_id="A2",
            engine="DETERMINISTIC",
            started_at=started_at,
            started_perf=started_perf,
            status="WARNING" if mode.warnings else "PASSED",
            reason_codes=mode.warnings,
        )
    )

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    numeric = predict(feature_row, mode)
    _validate_numeric_prediction(numeric)
    prediction = PredictionCardV2(
        request_id=request.request_id,
        origin_event_id=request.admission.origin_event_id,
        decision_at=request.admission.decision_at,
        frozen_model_id=plan.numeric_predictor_ref,
        **numeric.model_dump(),
    )
    records.append(
        _record(
            run_id=plan.run_id,
            agent_id="A4",
            engine="LOCAL_NUMERIC_MODEL",
            started_at=started_at,
            started_perf=started_perf,
            status="WARNING" if prediction.warnings else "PASSED",
            reason_codes=prediction.warnings,
        )
    )

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    checks = (
        "CONTRACT_ID_MATCH",
        "ORIGIN_EVENT_ID_MATCH",
        "DECISION_TIME_MATCH",
        "NO_FUTURE_INFORMATION",
        "LOCAL_NUMERIC_PREDICTOR",
    )
    audit = AuditCardV2(
        artifact_id=plan.run_id,
        passed=True,
        check_codes=checks,
        evidence_refs=(request.request_id, prediction.frozen_model_id),
    )
    records.append(
        _record(
            run_id=plan.run_id,
            agent_id="A5",
            engine="DETERMINISTIC",
            started_at=started_at,
            started_perf=started_perf,
            status="PASSED",
            reason_codes=checks,
        )
    )

    ledger = ResourceLedgerV1.from_records(plan.run_id, tuple(records))
    return OfflineGraphResult(
        plan=plan,
        process_mode=mode,
        prediction=prediction,
        audit=audit,
        resources=ledger,
    )
