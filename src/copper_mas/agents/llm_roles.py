"""Kimi 编排与 DeepSeek 说明任务的最小、不可越权合同。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from copper_mas.llm.adapters import StructuredTask


class OrchestrationDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["PROCEED", "HOLD"]
    optional_explanation_tasks: tuple[Literal["A2_EXPLANATION", "A5_SUMMARY"], ...] = ()
    reason_codes: tuple[str, ...] = Field(default=(), max_length=16)


class ExplanationCardV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    summary: str = Field(min_length=1, max_length=1200)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=32)
    changed_numeric_result: Literal[False] = False


def build_kimi_orchestration_task(
    *,
    artifact_ids: tuple[str, ...],
    pass_fail_status: str,
    reason_codes: tuple[str, ...],
    bounded_counts: dict[str, int | float | bool | str | None],
) -> StructuredTask:
    """Kimi 只能决定继续/暂停和是否请求非数值说明。"""

    return StructuredTask(
        task_name="orchestration_decision_v1",
        system_prompt=(
            "你是铜电解—电积研究工作流编排器。只根据合同化元数据返回JSON。"
            "不得要求原始数据，不得改变A1→A2→A4→A5顺序、模型、数值预测或审计结论。"
        ),
        user_payload={
            "artifact_ids": artifact_ids,
            "schema_version": "1.0",
            "pass_fail_status": pass_fail_status,
            "reason_codes": reason_codes,
            "bounded_aggregate_metadata": bounded_counts,
            "instruction": "决定固定工作流继续或暂停，并可请求A2/A5的非数值说明。",
            "requested_output": "OrchestrationDecisionV1",
        },
        output_schema=OrchestrationDecisionV1.model_json_schema(),
    )


def build_deepseek_explanation_task(
    *,
    artifact_ids: tuple[str, ...],
    pass_fail_status: str,
    reason_codes: tuple[str, ...],
    bounded_counts: dict[str, int | float | bool | str | None],
    explanation_scope: Literal["A2_MODE_EVIDENCE", "A5_AUDIT_SUMMARY"],
) -> StructuredTask:
    """DeepSeek 只把已有原因码整理成人类可读说明。"""

    return StructuredTask(
        task_name="explanation_card_v1",
        system_prompt=(
            "你是结构化说明执行器。只解释给定工件状态与原因码，不推测工厂原始值，"
            "不修改工况、模型、预测或审计结果，只返回JSON。"
        ),
        user_payload={
            "artifact_ids": artifact_ids,
            "schema_version": "1.0",
            "pass_fail_status": pass_fail_status,
            "reason_codes": reason_codes,
            "bounded_aggregate_metadata": {
                **bounded_counts,
                "explanation_scope": explanation_scope,
            },
            "instruction": "将已有证据整理为简短、可追溯、无新增数值判断的中文说明。",
            "requested_output": "ExplanationCardV1",
        },
        output_schema=ExplanationCardV1.model_json_schema(),
    )


def enforce_orchestration_decision(
    decision: OrchestrationDecisionV1,
    *,
    deterministic_failure_codes: tuple[str, ...],
) -> None:
    """确定性失败存在时，Kimi 无权把工作流改成继续。"""

    if deterministic_failure_codes and decision.action == "PROCEED":
        raise ValueError("Kimi 无权覆盖确定性安全/审计失败")
