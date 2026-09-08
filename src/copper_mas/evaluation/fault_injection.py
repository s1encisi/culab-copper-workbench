"""P3 开发期离线故障注入与安全消融。

该模块只验证预测路径的准入、审计和拒绝能力，不评价预测精度，也不读取
2026 外部时序留出集或目标真值账本。``direct_fixed_predictor`` 是有意保留的
无防护消融基线；``guarded_multi_agent_graph`` 在现有 A1→A2→A4→A5 运行时
外层增加原始合同、必需特征组、模型完整性和 LLM 边界检查。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import math
from time import perf_counter_ns
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError

from copper_mas.agents.runtime import NumericPrediction, fixed_offline_plan, run_offline_graph
from copper_mas.config.settings import ProviderRuntime
from copper_mas.contracts.cards import ForecastRequestV2
from copper_mas.data.leakage import FutureInformationError
from copper_mas.llm.adapters import StructuredTask
from copper_mas.llm.client import CallBudget, ExternalLLMCallsDisabled, call_structured


DIRECT_SYSTEM = "direct_fixed_predictor"
GUARDED_SYSTEM = "guarded_multi_agent_graph"
SINGLE_LLM_SYSTEM = "single_llm_agent"


class _LLMStatusEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["PASSED", "WARNING", "FAILED"]
    reason_codes: tuple[str, ...] = ()


class GuardBlocked(RuntimeError):
    """带稳定原因码的 fail-closed 结果。"""

    def __init__(self, reason_code: str, *, abstain: bool = False) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.abstain = abstain


@dataclass(frozen=True)
class FaultScenarioV1:
    scenario_id: str
    scenario_name_zh: str
    category_zh: str
    description_zh: str
    injected_fault_zh: str
    expected_guard_reason_code: str
    is_fault: bool
    request_payload: dict[str, Any]
    feature_row: dict[str, Any]
    predictor_mode: Literal["NORMAL", "NONFINITE", "INVALID_INTERVAL"] = "NORMAL"
    model_manifest: dict[str, str] | None = None
    model_artifact: bytes | None = None
    llm_mode: Literal["NONE", "LIVE_DISABLED", "INVALID_OUTPUT"] = "NONE"
    llm_output: dict[str, Any] | None = None

    def definition_row(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "场景名称": self.scenario_name_zh,
            "类别": self.category_zh,
            "是否故障": self.is_fault,
            "注入内容": self.injected_fault_zh,
            "验证目的": self.description_zh,
            "预期防护原因码": self.expected_guard_reason_code,
        }


@dataclass(frozen=True)
class FaultRunResultV1:
    repeat_index: int
    scenario_id: str
    scenario_name_zh: str
    system: str
    is_fault: bool
    prediction_emitted: bool
    unsafe_prediction_emitted: bool
    failure_detected: bool
    abstained_or_blocked: bool
    outcome_status: Literal["EMITTED", "BLOCKED", "ABSTAINED"]
    reason_code: str
    latency_ms: float
    llm_call_count: int = 0
    network_call_count: int = 0
    external_2026_read: bool = False
    outcome_ledger_read: bool = False

    def as_row(self) -> dict[str, Any]:
        return {
            "repeat_index": self.repeat_index,
            "scenario_id": self.scenario_id,
            "场景名称": self.scenario_name_zh,
            "system": self.system,
            "is_fault": self.is_fault,
            "prediction_emitted": self.prediction_emitted,
            "unsafe_prediction_emitted": self.unsafe_prediction_emitted,
            "failure_detected": self.failure_detected,
            "abstained_or_blocked": self.abstained_or_blocked,
            "outcome_status": self.outcome_status,
            "reason_code": self.reason_code,
            "latency_ms": self.latency_ms,
            "llm_call_count": self.llm_call_count,
            "network_call_count": self.network_call_count,
            "external_2026_read": self.external_2026_read,
            "outcome_ledger_read": self.outcome_ledger_read,
        }


def _base_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    decision_at = datetime(2025, 6, 1, 8, 0)
    origin_event_id = "development_synthetic_origin_001"
    request = {
        "request_id": "fault-injection-request-001",
        "admission": {
            "origin_event_id": origin_event_id,
            "decision_at": decision_at,
            "feature_cutoff_at": decision_at,
            "admission_status": "ADMITTED",
            "reason_codes": (),
            "feature_group_counts": {
                "current_waste_tank_result": 1,
                "phase_1": 7,
                "phase_2": 7,
                "stage34": 7,
            },
            "missing_feature_groups": (),
            "source_quality_warnings": (),
        },
        "observations": {
            "origin_event_id": origin_event_id,
            "decision_at": decision_at,
            "observations": (
                {
                    "canonical_tag": "origin_cu_g_l",
                    "value": 4.72,
                    "unit": "g/L",
                    "event_at": decision_at,
                    "available_at": decision_at,
                    "matched_anchor_at": decision_at,
                    "age_hours": 0.0,
                    "missing": False,
                    "quality_code": "SYNTHETIC_DEVELOPMENT_BENCHMARK",
                },
                {
                    "canonical_tag": "origin_as_mg_l",
                    "value": 5200.0,
                    "unit": "mg/L",
                    "event_at": decision_at,
                    "available_at": decision_at,
                    "matched_anchor_at": decision_at,
                    "age_hours": 0.0,
                    "missing": False,
                    "quality_code": "SYNTHETIC_DEVELOPMENT_BENCHMARK",
                },
            ),
        },
    }
    features = {
        "origin_cu_g_l": 4.72,
        "origin_as_mg_l": 5200.0,
        "phase1_stage12_current_ka__t_minus_0h": 10.0,
        "phase1_stage12_voltage_v__t_minus_0h": 20.0,
        "phase1_stage12_active_cells__t_minus_0h": 12.0,
        "phase2_stage12_current_ka__t_minus_0h": 15.0,
        "phase2_stage12_voltage_v__t_minus_0h": 19.0,
        "phase2_stage12_active_cells__t_minus_0h": 10.0,
        "stage3_current_a__t_minus_0h": 320.0,
        "stage3_voltage_v__t_minus_0h": 9.0,
        "stage3_flow_m3_h__t_minus_0h": 12.0,
        "stage4_current_a__t_minus_0h": 0.0,
        "stage4_voltage_v__t_minus_0h": 0.0,
        "stage4_flow_m3_h__t_minus_0h": 0.0,
    }
    return request, features


def build_fault_scenarios_v1() -> tuple[FaultScenarioV1, ...]:
    """生成一个正常对照和十个确定性故障场景。"""

    request, features = _base_payload()

    def make(
        scenario_id: str,
        scenario_name_zh: str,
        category_zh: str,
        injected_fault_zh: str,
        description_zh: str,
        reason_code: str,
        *,
        is_fault: bool = True,
        request_payload: dict[str, Any] | None = None,
        feature_row: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> FaultScenarioV1:
        return FaultScenarioV1(
            scenario_id=scenario_id,
            scenario_name_zh=scenario_name_zh,
            category_zh=category_zh,
            description_zh=description_zh,
            injected_fault_zh=injected_fault_zh,
            expected_guard_reason_code=reason_code,
            is_fault=is_fault,
            request_payload=deepcopy(request if request_payload is None else request_payload),
            feature_row=deepcopy(features if feature_row is None else feature_row),
            **kwargs,
        )

    scenarios: list[FaultScenarioV1] = [
        make(
            "S00_CLEAN_CONTROL",
            "正常合同对照",
            "对照",
            "不注入故障",
            "确认合法请求仍可由既有 A1→A2→A4→A5 图输出本地数值预测。",
            "PASSED_ALL_GUARDS",
            is_fault=False,
        )
    ]

    future_features = deepcopy(features)
    future_features["target_cu_g_l"] = 1.23
    scenarios.append(
        make(
            "S01_FUTURE_FIELD",
            "未来目标字段注入",
            "时间泄漏",
            "在 A4 特征行加入 target_cu_g_l",
            "验证任何 target_/next_/lead_to_ 字段在预测前被拒绝。",
            "FUTURE_INFORMATION_FORBIDDEN",
            feature_row=future_features,
        )
    )

    future_available_request = deepcopy(request)
    future_available_request["observations"]["observations"][0]["available_at"] = (
        request["admission"]["decision_at"] + timedelta(hours=2)
    )
    scenarios.append(
        make(
            "S02_AVAILABLE_AFTER_DECISION",
            "可用时间晚于决策时间",
            "As-Of 边界",
            "将一条观测 available_at 设为 decision_at+2h",
            "验证未来才可获得的观测不能进入预测请求。",
            "AVAILABLE_AFTER_DECISION",
            request_payload=future_available_request,
        )
    )

    rejected_request = deepcopy(request)
    rejected_request["admission"]["admission_status"] = "REJECTED"
    rejected_request["admission"]["reason_codes"] = ("SYNTHETIC_REJECTED",)
    scenarios.append(
        make(
            "S03_REJECTED_ADMISSION",
            "REJECTED 准入仍请求预测",
            "准入状态",
            "将 admission_status 设为 REJECTED 后继续发起预测",
            "验证被 A1 拒绝的请求不能绕过准入进入 A2/A4。",
            "ADMISSION_REJECTED",
            request_payload=rejected_request,
        )
    )

    mismatch_request = deepcopy(request)
    mismatch_request["observations"]["origin_event_id"] = "different-origin-event"
    mismatch_request["observations"]["decision_at"] = (
        request["admission"]["decision_at"] + timedelta(minutes=1)
    )
    scenarios.append(
        make(
            "S04_ID_TIME_MISMATCH",
            "事件 ID 与时间不一致",
            "卡片一致性",
            "观察卡的 origin_event_id 与 decision_at 同时偏离准入卡",
            "验证跨卡片身份与时间必须完全一致。",
            "REQUEST_IDENTITY_TIME_MISMATCH",
            request_payload=mismatch_request,
        )
    )

    missing_stage34_request = deepcopy(request)
    missing_stage34_request["admission"]["admission_status"] = "WARNING"
    missing_stage34_request["admission"]["feature_group_counts"]["stage34"] = 0
    missing_stage34_request["admission"]["missing_feature_groups"] = ("stage34",)
    missing_stage34_request["admission"]["source_quality_warnings"] = (
        "CORE_PROCESS_GROUP_LT_4_OF_7:stage34",
    )
    scenarios.append(
        make(
            "S05_STAGE34_GROUP_MISSING",
            "三四段整组缺失",
            "数据完整性",
            "将 stage34 组计数置零并登记为缺失组",
            "验证核心工艺组完全缺失时采取弃权，不冒险给出数字。",
            "REQUIRED_STAGE34_GROUP_MISSING",
            request_payload=missing_stage34_request,
        )
    )

    scenarios.append(
        make(
            "S06_NONFINITE_PREDICTION",
            "非有限数值预测",
            "数值输出",
            "令 A4 输出 Cu=NaN",
            "验证 NaN/Inf 不会被封装成正式预测卡。",
            "NONFINITE_NUMERIC_PREDICTION",
            predictor_mode="NONFINITE",
        )
    )
    scenarios.append(
        make(
            "S07_INVALID_INTERVAL",
            "无效预测区间",
            "数值输出",
            "令 Cu 点预测为 4.72，但区间为 [6,7]",
            "验证区间边界完整、有限且必须覆盖点预测。",
            "INVALID_PREDICTION_INTERVAL",
            predictor_mode="INVALID_INTERVAL",
        )
    )

    artifact = b"local-development-model-artifact-v1"
    scenarios.append(
        make(
            "S08_MODEL_HASH_TAMPER",
            "模型清单与哈希篡改",
            "模型供应链",
            "模型清单 model_id 被替换且预期 SHA-256 与本地字节不符",
            "验证 A4 加载前必须核对冻结模型标识和文件哈希。",
            "MODEL_MANIFEST_OR_HASH_MISMATCH",
            model_manifest={
                "model_id": "TAMPERED_MODEL_ID",
                "expected_model_id": "PERSISTENCE_CURRENT_RESULT_V1",
                "sha256": "0" * 64,
            },
            model_artifact=artifact,
        )
    )

    scenarios.append(
        make(
            "S09_LLM_LIVE_DISABLED",
            "关闭真实调用时请求 LLM",
            "外部调用边界",
            "在 live_calls=false 下提交结构化 LLM 请求",
            "验证在建立网络连接前拒绝调用，调用数和网络数均为零。",
            "LLM_LIVE_CALLS_DISABLED",
            llm_mode="LIVE_DISABLED",
        )
    )
    scenarios.append(
        make(
            "S10_LLM_SCHEMA_INVALID",
            "非法 LLM 结构化输出",
            "LLM 输出合同",
            "本地注入 status=MAYBE、reason_codes 为字符串且含额外字段",
            "验证 LLM 文本不能绕过本地 Pydantic 合同进入工件链。",
            "LLM_OUTPUT_SCHEMA_INVALID",
            llm_mode="INVALID_OUTPUT",
            llm_output={
                "status": "MAYBE",
                "reason_codes": "not-an-array",
                "unexpected": True,
            },
        )
    )
    return tuple(scenarios)


def _numeric_for(scenario: FaultScenarioV1) -> NumericPrediction:
    cu = float(scenario.feature_row.get("origin_cu_g_l", 4.72))
    arsenic = float(scenario.feature_row.get("origin_as_mg_l", 5200.0))
    if scenario.predictor_mode == "NONFINITE":
        return NumericPrediction(predicted_cu_g_l=math.nan, predicted_as_mg_l=arsenic)
    if scenario.predictor_mode == "INVALID_INTERVAL":
        return NumericPrediction(
            predicted_cu_g_l=cu,
            predicted_as_mg_l=arsenic,
            cu_interval_low=6.0,
            cu_interval_high=7.0,
        )
    return NumericPrediction(predicted_cu_g_l=cu, predicted_as_mg_l=arsenic)


def _verify_model_integrity(scenario: FaultScenarioV1) -> None:
    if scenario.model_manifest is None:
        return
    if scenario.model_artifact is None:
        raise GuardBlocked("MODEL_MANIFEST_OR_HASH_MISMATCH")
    manifest = scenario.model_manifest
    actual_hash = hashlib.sha256(scenario.model_artifact).hexdigest()
    if (
        manifest.get("model_id") != manifest.get("expected_model_id")
        or manifest.get("sha256") != actual_hash
    ):
        raise GuardBlocked("MODEL_MANIFEST_OR_HASH_MISMATCH")


def _safe_llm_task() -> StructuredTask:
    return StructuredTask(
        task_name="fault_injection_status",
        system_prompt="只返回合同允许的 JSON。",
        user_payload={
            "artifact_ids": ["p3-fault-injection"],
            "schema_version": "1.0",
            "pass_fail_status": "FAILED",
            "reason_codes": ["SYNTHETIC_FAULT"],
            "bounded_aggregate_metadata": {"scenario_count": 1},
            "instruction": "汇总离线故障状态。",
            "requested_output": "status and reason_codes",
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"enum": ["PASSED", "WARNING", "FAILED"]},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "reason_codes"],
            "additionalProperties": False,
        },
    )


def _enforce_optional_boundaries(scenario: FaultScenarioV1) -> None:
    _verify_model_integrity(scenario)
    if scenario.llm_mode == "LIVE_DISABLED":
        runtime = ProviderRuntime(
            name="deepseek_executor",
            live_calls=False,
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url=None,
            api_key=None,
            reasoning_effort="medium",
            max_completion_tokens=64,
            max_calls_per_run=1,
        )
        try:
            call_structured(
                runtime=runtime,
                task=_safe_llm_task(),
                output_model=_LLMStatusEnvelope,
                budget=CallBudget(1),
            )
        except ExternalLLMCallsDisabled as exc:
            raise GuardBlocked("LLM_LIVE_CALLS_DISABLED") from exc
        raise AssertionError("live_calls=false 不应完成外部调用")
    if scenario.llm_mode == "INVALID_OUTPUT":
        try:
            _LLMStatusEnvelope.model_validate(scenario.llm_output)
        except ValidationError as exc:
            raise GuardBlocked("LLM_OUTPUT_SCHEMA_INVALID") from exc


def _required_group_guard(request: ForecastRequestV2) -> None:
    admission = request.admission
    if (
        "stage34" in admission.missing_feature_groups
        or admission.feature_group_counts.get("stage34", 0) <= 0
    ):
        raise GuardBlocked("REQUIRED_STAGE34_GROUP_MISSING", abstain=True)


def _classify_validation_error(exc: ValidationError) -> str:
    message = str(exc)
    if "REJECTED" in message:
        return "ADMISSION_REJECTED"
    if "origin_event_id" in message or "decision_at" in message:
        if "之后才可用" in message:
            return "AVAILABLE_AFTER_DECISION"
        return "REQUEST_IDENTITY_TIME_MISMATCH"
    return "REQUEST_CONTRACT_INVALID"


def _classify_value_error(exc: ValueError) -> str:
    message = str(exc)
    if "非有限" in message:
        return "NONFINITE_NUMERIC_PREDICTION"
    if "区间" in message:
        return "INVALID_PREDICTION_INTERVAL"
    return "GUARD_VALUE_ERROR"


def run_direct_fixed_predictor(
    scenario: FaultScenarioV1, *, repeat_index: int
) -> FaultRunResultV1:
    """有意不做合同/完整性检查的消融基线。"""

    started = perf_counter_ns()
    _numeric_for(scenario)
    latency_ms = (perf_counter_ns() - started) / 1_000_000
    return FaultRunResultV1(
        repeat_index=repeat_index,
        scenario_id=scenario.scenario_id,
        scenario_name_zh=scenario.scenario_name_zh,
        system=DIRECT_SYSTEM,
        is_fault=scenario.is_fault,
        prediction_emitted=True,
        unsafe_prediction_emitted=scenario.is_fault,
        failure_detected=False,
        abstained_or_blocked=False,
        outcome_status="EMITTED",
        reason_code="NO_SAFETY_GUARD",
        latency_ms=latency_ms,
    )


def run_guarded_multi_agent_graph(
    scenario: FaultScenarioV1, *, repeat_index: int
) -> FaultRunResultV1:
    """运行原始合同校验、外层防护和既有离线图，所有错误均 fail-closed。"""

    started = perf_counter_ns()
    emitted = False
    detected = False
    outcome: Literal["EMITTED", "BLOCKED", "ABSTAINED"] = "BLOCKED"
    reason_code = "UNCLASSIFIED_GUARD_FAILURE"
    try:
        request = ForecastRequestV2.model_validate(scenario.request_payload)
        _required_group_guard(request)
        _enforce_optional_boundaries(scenario)
        plan = fixed_offline_plan(
            run_id=f"p3-{scenario.scenario_id}-{repeat_index}",
            numeric_predictor_ref="P3_SYNTHETIC_LOCAL_PREDICTOR_V1",
        )
        result = run_offline_graph(
            plan=plan,
            request=request,
            feature_row=scenario.feature_row,
            predict=lambda _row, _mode: _numeric_for(scenario),
        )
        if not result.audit.passed or result.resources.total_calls != 0:
            raise GuardBlocked("A5_AUDIT_FAILED")
        emitted = True
        outcome = "EMITTED"
        reason_code = "PASSED_ALL_GUARDS"
    except GuardBlocked as exc:
        detected = True
        outcome = "ABSTAINED" if exc.abstain else "BLOCKED"
        reason_code = exc.reason_code
    except FutureInformationError as exc:
        detected = True
        outcome = "BLOCKED"
        reason_code = (
            "AVAILABLE_AFTER_DECISION"
            if "available_at" in str(exc)
            else "FUTURE_INFORMATION_FORBIDDEN"
        )
    except ValidationError as exc:
        detected = True
        outcome = "BLOCKED"
        reason_code = _classify_validation_error(exc)
    except ValueError as exc:
        detected = True
        outcome = "BLOCKED"
        reason_code = _classify_value_error(exc)

    latency_ms = (perf_counter_ns() - started) / 1_000_000
    return FaultRunResultV1(
        repeat_index=repeat_index,
        scenario_id=scenario.scenario_id,
        scenario_name_zh=scenario.scenario_name_zh,
        system=GUARDED_SYSTEM,
        is_fault=scenario.is_fault,
        prediction_emitted=emitted,
        unsafe_prediction_emitted=emitted and scenario.is_fault,
        failure_detected=detected,
        abstained_or_blocked=outcome in {"BLOCKED", "ABSTAINED"},
        outcome_status=outcome,
        reason_code=reason_code,
        latency_ms=latency_ms,
    )


def execute_fault_benchmark_v1(
    *, repeats: int = 100, warmup_repeats: int = 3
) -> tuple[tuple[FaultScenarioV1, ...], tuple[FaultRunResultV1, ...]]:
    if repeats < 2:
        raise ValueError("至少重复 2 次才能报告 p50/p95")
    if warmup_repeats < 0:
        raise ValueError("warmup_repeats 不能为负数")
    scenarios = build_fault_scenarios_v1()
    for scenario in scenarios:
        for index in range(warmup_repeats):
            run_direct_fixed_predictor(scenario, repeat_index=-(index + 1))
            run_guarded_multi_agent_graph(scenario, repeat_index=-(index + 1))

    results: list[FaultRunResultV1] = []
    for repeat_index in range(1, repeats + 1):
        for scenario in scenarios:
            # 同一重复内交替执行次序，减小固定先后顺序对微秒级延迟的偏差。
            if repeat_index % 2:
                results.append(run_direct_fixed_predictor(scenario, repeat_index=repeat_index))
                results.append(
                    run_guarded_multi_agent_graph(scenario, repeat_index=repeat_index)
                )
            else:
                results.append(
                    run_guarded_multi_agent_graph(scenario, repeat_index=repeat_index)
                )
                results.append(run_direct_fixed_predictor(scenario, repeat_index=repeat_index))
    return scenarios, tuple(results)


def assert_benchmark_expectations(
    scenarios: tuple[FaultScenarioV1, ...], results: tuple[FaultRunResultV1, ...]
) -> None:
    expected_reason = {item.scenario_id: item.expected_guard_reason_code for item in scenarios}
    for row in results:
        if row.system == DIRECT_SYSTEM:
            if not row.prediction_emitted or row.failure_detected or row.abstained_or_blocked:
                raise AssertionError(f"直接基线行为偏离消融定义: {row}")
            if row.unsafe_prediction_emitted != row.is_fault:
                raise AssertionError(f"直接基线危险输出标记不一致: {row}")
            continue
        if row.system != GUARDED_SYSTEM:
            raise AssertionError(f"未知系统: {row.system}")
        if row.reason_code != expected_reason[row.scenario_id]:
            raise AssertionError(f"防护原因码不符合场景定义: {row}")
        if row.is_fault:
            if row.prediction_emitted or row.unsafe_prediction_emitted:
                raise AssertionError(f"防护图在故障场景输出了预测: {row}")
            if not row.failure_detected or not row.abstained_or_blocked:
                raise AssertionError(f"防护图未发现或未拒绝故障: {row}")
        elif not row.prediction_emitted or row.failure_detected:
            raise AssertionError(f"防护图错误拒绝正常对照: {row}")


def clone_with_feature_row(
    scenario: FaultScenarioV1, feature_row: Mapping[str, Any]
) -> FaultScenarioV1:
    """测试辅助函数：保留场景定义，仅替换特征行。"""

    return replace(scenario, feature_row=dict(feature_row))
