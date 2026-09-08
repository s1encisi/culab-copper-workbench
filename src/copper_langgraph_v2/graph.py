"""A1→A2→A4→A5 的 LangGraph-only 实现；无 LLM、无网络、无 2026 读取。"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from copper_langgraph_v2.bridge import (
    AgentExecutionRecordV1,
    AuditCardV2,
    ForecastRequestV2,
    NumericPrediction,
    PredictFunction,
    PredictionCardV2,
    ProcessModeCardV2,
    ResourceLedgerV1,
    WorkflowPlanV1,
    _record,
    _validate_numeric_prediction,
    assert_no_future_information,
    infer_process_mode,
)


class RunIdConflictError(RuntimeError):
    """相同 run_id 被用于不同输入。"""


class GraphState(TypedDict, total=False):
    plan: dict[str, Any]
    request: dict[str, Any]
    feature_row: dict[str, Any]
    input_fingerprint: str
    trace: list[str]
    records: list[dict[str, Any]]
    process_mode: dict[str, Any] | None
    prediction: dict[str, Any] | None
    audit: dict[str, Any] | None
    resources: dict[str, Any] | None
    failure: dict[str, str] | None
    blocked: bool
    completed: bool
    idempotent_replay: bool


def persistence_predictor(
    feature_row: Mapping[str, Any], process_mode: ProcessModeCardV2
) -> NumericPrediction:
    """与 V1 冻结 Persistence 相同的本地数值公式。"""

    del process_mode
    return NumericPrediction(
        predicted_cu_g_l=float(feature_row["origin_cu_g_l"]),
        predicted_as_mg_l=float(feature_row["origin_as_mg_l"]),
    )


def _fingerprint(
    *, plan: WorkflowPlanV1, request: ForecastRequestV2, feature_row: Mapping[str, Any]
) -> str:
    payload = {
        "plan": plan.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
        "feature_row": _normalize_state_value(dict(feature_row)),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_state_value(value: Any) -> Any:
    """把 pandas/numpy 标量转成 SQLite checkpointer 可稳定序列化的值。"""

    if isinstance(value, Mapping):
        return {str(key): _normalize_state_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_state_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_state_value(item) for item in value]
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return _normalize_state_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _append_record(
    state: GraphState,
    *,
    agent_id: str,
    engine: str,
    started_at: datetime,
    started_perf: float,
    status: str,
    reason_codes: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    plan = WorkflowPlanV1.model_validate(state["plan"])
    record = _record(
        run_id=plan.run_id,
        agent_id=agent_id,
        engine=engine,
        started_at=started_at,
        started_perf=started_perf,
        status=status,
        reason_codes=reason_codes,
    )
    return [*state.get("records", []), record.model_dump(mode="json")]


def _failure(
    state: GraphState,
    *,
    node: str,
    engine: str,
    started_at: datetime,
    started_perf: float,
    exc: Exception,
) -> dict[str, Any]:
    code = f"{node}_{type(exc).__name__.upper()}"
    records = _append_record(
        state,
        agent_id=node,
        engine=engine,
        started_at=started_at,
        started_perf=started_perf,
        status="FAILED",
        reason_codes=(code,),
    )
    return {
        "records": records,
        "trace": [*state.get("trace", []), node],
        "failure": {"node": node, "code": code, "message": str(exc)},
    }


def _route(state: GraphState) -> str:
    return "fail" if state.get("failure") else "continue"


def _build_graph(*, predict: PredictFunction, checkpointer: SqliteSaver):
    def a1(state: GraphState) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started_perf = perf_counter()
        try:
            plan = WorkflowPlanV1.model_validate(state["plan"])
            request = ForecastRequestV2.model_validate(state["request"])
            if plan.run_id != state["plan"]["run_id"]:
                raise ValueError("run_id 不一致")
            assert_no_future_information(
                request, decision_at=request.admission.decision_at
            )
            assert_no_future_information(
                state["feature_row"], decision_at=request.admission.decision_at
            )
            records = _append_record(
                state,
                agent_id="A1",
                engine="DETERMINISTIC",
                started_at=started_at,
                started_perf=started_perf,
                status=(
                    "WARNING"
                    if request.admission.admission_status == "WARNING"
                    else "PASSED"
                ),
                reason_codes=request.admission.reason_codes,
            )
            return {
                "records": records,
                "trace": [*state.get("trace", []), "A1"],
                "failure": None,
            }
        except Exception as exc:  # fail-closed boundary
            return _failure(
                state,
                node="A1",
                engine="DETERMINISTIC",
                started_at=started_at,
                started_perf=started_perf,
                exc=exc,
            )

    def a2(state: GraphState) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started_perf = perf_counter()
        try:
            request = ForecastRequestV2.model_validate(state["request"])
            mode = infer_process_mode(
                origin_event_id=request.admission.origin_event_id,
                decision_at=request.admission.decision_at,
                feature_row=state["feature_row"],
            )
            records = _append_record(
                state,
                agent_id="A2",
                engine="DETERMINISTIC",
                started_at=started_at,
                started_perf=started_perf,
                status="WARNING" if mode.warnings else "PASSED",
                reason_codes=mode.warnings,
            )
            return {
                "process_mode": mode.model_dump(mode="json"),
                "records": records,
                "trace": [*state.get("trace", []), "A2"],
                "failure": None,
            }
        except Exception as exc:
            return _failure(
                state,
                node="A2",
                engine="DETERMINISTIC",
                started_at=started_at,
                started_perf=started_perf,
                exc=exc,
            )

    def a4(state: GraphState) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started_perf = perf_counter()
        try:
            plan = WorkflowPlanV1.model_validate(state["plan"])
            request = ForecastRequestV2.model_validate(state["request"])
            mode = ProcessModeCardV2.model_validate(state["process_mode"])
            numeric = predict(state["feature_row"], mode)
            _validate_numeric_prediction(numeric)
            prediction = PredictionCardV2(
                request_id=request.request_id,
                origin_event_id=request.admission.origin_event_id,
                decision_at=request.admission.decision_at,
                frozen_model_id=plan.numeric_predictor_ref,
                **numeric.model_dump(),
            )
            records = _append_record(
                state,
                agent_id="A4",
                engine="LOCAL_NUMERIC_MODEL",
                started_at=started_at,
                started_perf=started_perf,
                status="WARNING" if prediction.warnings else "PASSED",
                reason_codes=prediction.warnings,
            )
            return {
                "prediction": prediction.model_dump(mode="json"),
                "records": records,
                "trace": [*state.get("trace", []), "A4"],
                "failure": None,
            }
        except Exception as exc:
            return _failure(
                state,
                node="A4",
                engine="LOCAL_NUMERIC_MODEL",
                started_at=started_at,
                started_perf=started_perf,
                exc=exc,
            )

    def a5(state: GraphState) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started_perf = perf_counter()
        try:
            plan = WorkflowPlanV1.model_validate(state["plan"])
            request = ForecastRequestV2.model_validate(state["request"])
            prediction = PredictionCardV2.model_validate(state["prediction"])
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
            records = _append_record(
                state,
                agent_id="A5",
                engine="DETERMINISTIC",
                started_at=started_at,
                started_perf=started_perf,
                status="PASSED",
                reason_codes=checks,
            )
            ledger = ResourceLedgerV1.from_records(
                plan.run_id,
                tuple(AgentExecutionRecordV1.model_validate(item) for item in records),
            )
            return {
                "audit": audit.model_dump(mode="json"),
                "resources": ledger.model_dump(mode="json"),
                "records": records,
                "trace": [*state.get("trace", []), "A5"],
                "failure": None,
                "blocked": False,
                "completed": True,
            }
        except Exception as exc:
            return _failure(
                state,
                node="A5",
                engine="DETERMINISTIC",
                started_at=started_at,
                started_perf=started_perf,
                exc=exc,
            )

    def fail_closed(state: GraphState) -> dict[str, Any]:
        plan = WorkflowPlanV1.model_validate(state["plan"])
        failure = state.get("failure") or {
            "node": "UNKNOWN",
            "code": "UNKNOWN_FAILURE",
            "message": "未知失败",
        }
        audit = AuditCardV2(
            artifact_id=plan.run_id,
            passed=False,
            check_codes=(),
            failure_codes=(failure["code"],),
            evidence_refs=(),
        )
        records = tuple(
            AgentExecutionRecordV1.model_validate(item)
            for item in state.get("records", [])
        )
        ledger = ResourceLedgerV1.from_records(plan.run_id, records)
        return {
            "prediction": None,
            "audit": audit.model_dump(mode="json"),
            "resources": ledger.model_dump(mode="json"),
            "trace": [*state.get("trace", []), "FAIL_CLOSED"],
            "blocked": True,
            "completed": True,
        }

    builder = StateGraph(GraphState)
    builder.add_node("A1", a1)
    builder.add_node("A2", a2)
    builder.add_node("A4", a4)
    builder.add_node("A5", a5)
    builder.add_node("FAIL_CLOSED", fail_closed)
    builder.add_edge(START, "A1")
    builder.add_conditional_edges(
        "A1", _route, {"continue": "A2", "fail": "FAIL_CLOSED"}
    )
    builder.add_conditional_edges(
        "A2", _route, {"continue": "A4", "fail": "FAIL_CLOSED"}
    )
    builder.add_conditional_edges(
        "A4", _route, {"continue": "A5", "fail": "FAIL_CLOSED"}
    )
    builder.add_conditional_edges(
        "A5", _route, {"continue": END, "fail": "FAIL_CLOSED"}
    )
    builder.add_edge("FAIL_CLOSED", END)
    return builder.compile(checkpointer=checkpointer)


class LangGraphRunner:
    """SQLite 检查点运行器；相同 run_id + 相同输入只返回既有结果。"""

    def __init__(self, *, checkpoint_path: str | Path, predict: PredictFunction):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.checkpoint_path, check_same_thread=False
        )
        self._checkpointer = SqliteSaver(self._connection)
        self._checkpointer.setup()
        self.graph = _build_graph(predict=predict, checkpointer=self._checkpointer)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "LangGraphRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _config(run_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": run_id}}

    def history_count(self, run_id: str) -> int:
        return sum(1 for _ in self.graph.get_state_history(self._config(run_id)))

    def invoke(
        self,
        *,
        plan: WorkflowPlanV1,
        request: ForecastRequestV2,
        feature_row: Mapping[str, Any],
    ) -> GraphState:
        fingerprint = _fingerprint(
            plan=plan, request=request, feature_row=feature_row
        )
        config = self._config(plan.run_id)
        existing = dict(self.graph.get_state(config).values or {})
        if existing:
            if existing.get("input_fingerprint") != fingerprint:
                raise RunIdConflictError(
                    "相同 run_id 已绑定不同输入；为防止重复副作用，已拒绝运行"
                )
            if existing.get("completed") is True:
                existing["idempotent_replay"] = True
                return existing  # type: ignore[return-value]
            raise RunIdConflictError("相同 run_id 存在未完成检查点，需人工恢复")

        initial: GraphState = {
            "plan": plan.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "feature_row": _normalize_state_value(dict(feature_row)),
            "input_fingerprint": fingerprint,
            "trace": [],
            "records": [],
            "process_mode": None,
            "prediction": None,
            "audit": None,
            "resources": None,
            "failure": None,
            "blocked": False,
            "completed": False,
            "idempotent_replay": False,
        }
        return self.graph.invoke(initial, config=config)
