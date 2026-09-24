"""Kimi/DeepSeek 首次连通的最小、脱敏、可计费预算工作流。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from time import perf_counter

import httpx
from pydantic import BaseModel, ConfigDict, Field

from copper_mas.config.settings import (
    BudgetRuntime,
    ProviderRegistry,
    resolve_provider_runtime,
)
from copper_mas.contracts.runtime import AgentExecutionRecordV1, ResourceLedgerV1
from copper_mas.llm.adapters import StructuredTask
from copper_mas.llm.client import CallBudget, call_structured

CONNECTIVITY_MAX_COMPLETION_TOKENS = 128
CONNECTIVITY_PROVIDERS = ("kimi_orchestrator", "deepseek_executor")


class ConnectivityReplyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = Field(pattern=r"^OK$")


def synthetic_connectivity_task_v1() -> StructuredTask:
    """仅包含合成元数据，不包含工厂行、Cu/As 数值或任何 2026 对象。"""

    return StructuredTask(
        task_name="connectivity_v1",
        system_prompt=("这是脱敏连通检查。不执行工艺推理，不调用工具，仅按 JSON Schema 返回 status=OK。"),
        user_payload={
            "artifact_ids": ["synthetic-connectivity-artifact-v1"],
            "schema_version": "1.0",
            "pass_fail_status": "SYNTHETIC_PREFLIGHT",
            "reason_codes": ["NO_PLANT_DATA", "NO_TARGET_DATA", "NO_SECRETS"],
            "bounded_aggregate_metadata": {
                "synthetic_record_count": 0,
                "contains_plant_data": False,
                "contains_2026_outcomes": False,
            },
            "instruction": "返回脱敏连通状态。",
            "requested_output": "status only",
        },
        output_schema={
            "type": "object",
            "properties": {"status": {"const": "OK"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )


def _effective_run_limit(budget: BudgetRuntime, *, declared_month_spend_cny: float) -> float:
    if declared_month_spend_cny < 0:
        raise ValueError("declared_month_spend_cny 不得为负")
    if budget.monthly_cny_limit is None:
        return budget.max_cny_per_run
    remaining = budget.monthly_cny_limit - declared_month_spend_cny
    if remaining <= 0:
        raise ValueError("申报的本月累计支出已达到月度预算上限")
    return min(budget.max_cny_per_run, remaining)


def run_connectivity_v1(
    *,
    registry: ProviderRegistry,
    budget_runtime: BudgetRuntime,
    environ: dict[str, str],
    execute_live: bool,
    run_id: str,
    provider_names: Sequence[str] = CONNECTIVITY_PROVIDERS,
    declared_month_spend_cny: float = 0.0,
    transports: Mapping[str, httpx.BaseTransport] | None = None,
) -> dict[str, object]:
    """运行或预检连通。默认 execute_live=False，不建立网络连接。"""

    selected = tuple(provider_names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("provider_names 必须是非空且不重复的序列")
    if any(name not in CONNECTIVITY_PROVIDERS for name in selected):
        raise ValueError("连通检查只允许 Kimi 和 DeepSeek 两个冻结供应商")
    effective_limit = _effective_run_limit(budget_runtime, declared_month_spend_cny=declared_month_spend_cny)

    runtime_env = dict(environ)
    runtime_env["COPPER_MAS_LIVE_CALLS"] = "true" if execute_live else "false"
    runtimes = {
        name: resolve_provider_runtime(registry, name, environ=runtime_env).model_copy(
            update={
                "max_completion_tokens": CONNECTIVITY_MAX_COMPLETION_TOKENS,
                "max_calls_per_run": 1,
            }
        )
        for name in selected
    }
    provider_report: dict[str, object] = {
        name: {
            "status": "READY_FOR_EXPLICIT_LIVE_RUN" if not execute_live else "PENDING",
            "provider": runtime.provider,
            "model": runtime.model,
            "base_url": runtime.base_url,
            "api_key_configured": runtime.api_key is not None,
            "max_calls_this_connectivity_run": 1,
            "max_completion_tokens": runtime.max_completion_tokens,
            "pricing_basis": runtime.pricing_basis,
            "input_cny_per_million_tokens": runtime.input_cny_per_million_tokens,
            "output_cny_per_million_tokens": runtime.output_cny_per_million_tokens,
        }
        for name, runtime in runtimes.items()
    }

    if not execute_live:
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "execution_status": "DRY_RUN_NO_NETWORK",
            "payload_policy": "SYNTHETIC_DEIDENTIFIED_METADATA_ONLY",
            "run_cost_limit_cny": effective_limit,
            "monthly_cny_limit": budget_runtime.monthly_cny_limit,
            "declared_month_spend_cny": declared_month_spend_cny,
            "providers": provider_report,
            "resource_ledger": None,
            "secret_values_printed": False,
        }

    call_budget = CallBudget(
        max_calls=len(selected),
        max_cost_cny=effective_limit,
        provider_call_limits=dict.fromkeys(selected, 1),
    )
    records: list[AgentExecutionRecordV1] = []
    task = synthetic_connectivity_task_v1()
    for name in selected:
        runtime = runtimes[name]
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        result = call_structured(
            runtime=runtime,
            task=task,
            output_model=ConnectivityReplyV1,
            budget=call_budget,
            transport=(transports or {}).get(name),
            timeout_seconds=30.0,
        )
        ended_at = datetime.now(UTC)
        record = AgentExecutionRecordV1(
            run_id=run_id,
            agent_id="ORCHESTRATOR",
            engine="KIMI" if runtime.provider == "moonshot" else "DEEPSEEK",
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=max(0.0, (perf_counter() - started_perf) * 1000),
            status="PASSED",
            call_count=1,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost_cny=result.estimated_cost_cny,
            cost_estimation_status=result.cost_status,
            reason_codes=("SYNTHETIC_CONNECTIVITY_OK",),
        )
        records.append(record)
        provider_report[name] = {
            **provider_report[name],
            "status": "PASSED",
            "response_status": result.value.status,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "estimated_cost_cny": result.estimated_cost_cny,
            "cost_status": result.cost_status,
        }

    ledger = ResourceLedgerV1.from_records(
        run_id,
        tuple(records),
        run_cost_limit_cny=effective_limit,
    )
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "execution_status": "LIVE_SYNTHETIC_CONNECTIVITY_COMPLETED",
        "payload_policy": "SYNTHETIC_DEIDENTIFIED_METADATA_ONLY",
        "run_cost_limit_cny": effective_limit,
        "monthly_cny_limit": budget_runtime.monthly_cny_limit,
        "declared_month_spend_cny": declared_month_spend_cny,
        "providers": provider_report,
        "resource_ledger": ledger.model_dump(mode="json"),
        "secret_values_printed": False,
    }
