"""三种工作流对照实验共用的资源与稳定性指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from copper_mas.contracts.runtime import ResourceLedgerV1

ArmName = Literal[
    "DETERMINISTIC_FIXED_WORKFLOW",
    "SINGLE_LLM_AGENT",
    "MULTI_AGENT_GRAPH",
]


@dataclass(frozen=True)
class ArmRunOutcome:
    arm: ArmName
    run_id: str
    passed: bool
    abstained: bool
    resources: ResourceLedgerV1


def summarize_arm_resources(outcomes: list[ArmRunOutcome]) -> dict[str, float | int | str]:
    """聚合 p50/p95、调用、token、成本、失败与拒绝率。"""

    if not outcomes:
        raise ValueError("至少需要一条运行结果")
    arms = {item.arm for item in outcomes}
    if len(arms) != 1:
        raise ValueError("一次汇总只能包含一个实验臂")
    latencies = np.asarray([item.resources.total_latency_ms for item in outcomes], dtype=float)
    count = len(outcomes)
    return {
        "arm": next(iter(arms)),
        "run_count": count,
        "passed_count": sum(item.passed for item in outcomes),
        "failure_count": sum(not item.passed for item in outcomes),
        "abstention_count": sum(item.abstained for item in outcomes),
        "failure_rate": sum(not item.passed for item in outcomes) / count,
        "abstention_rate": sum(item.abstained for item in outcomes) / count,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total_calls": sum(item.resources.total_calls for item in outcomes),
        "total_input_tokens": sum(item.resources.total_input_tokens for item in outcomes),
        "total_output_tokens": sum(item.resources.total_output_tokens for item in outcomes),
        "total_estimated_cost_cny": sum(item.resources.total_estimated_cost_cny for item in outcomes),
    }
