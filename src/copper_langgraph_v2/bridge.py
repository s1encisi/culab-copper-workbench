"""只读复用已经消费的 V1 本地合同与确定性函数。"""

from __future__ import annotations

from copper_langgraph_v2.paths import PROJECT_ROOT

# 保留旧名称供现有调用方兼容；V1 与 V2 现在位于同一正式工程根目录。
V1_PROJECT_ROOT = PROJECT_ROOT

from copper_mas.agents.mode import infer_process_mode
from copper_mas.agents.runtime import (
    NumericPrediction,
    OfflineGraphResult,
    PredictFunction,
    _record,
    _validate_numeric_prediction,
    fixed_offline_plan,
    run_offline_graph,
)
from copper_mas.contracts.cards import (
    AsOfAdmissionCardV2,
    AuditCardV2,
    ForecastRequestV2,
    ObservationCardV2,
    ObservationItemV2,
    PredictionCardV2,
    ProcessModeCardV2,
)
from copper_mas.contracts.runtime import (
    AgentExecutionRecordV1,
    ResourceLedgerV1,
    WorkflowPlanV1,
)
from copper_mas.data.leakage import assert_no_future_information

__all__ = [
    "AgentExecutionRecordV1",
    "AsOfAdmissionCardV2",
    "AuditCardV2",
    "ForecastRequestV2",
    "NumericPrediction",
    "ObservationCardV2",
    "ObservationItemV2",
    "OfflineGraphResult",
    "PredictFunction",
    "PredictionCardV2",
    "ProcessModeCardV2",
    "ResourceLedgerV1",
    "V1_PROJECT_ROOT",
    "WorkflowPlanV1",
    "_record",
    "_validate_numeric_prediction",
    "assert_no_future_information",
    "fixed_offline_plan",
    "infer_process_mode",
    "run_offline_graph",
]
