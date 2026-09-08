from datetime import datetime

import pytest

from copper_mas.agents.runtime import NumericPrediction, fixed_offline_plan, run_offline_graph
from copper_mas.contracts.cards import (
    AsOfAdmissionCardV2,
    ForecastRequestV2,
    ObservationCardV2,
    ObservationItemV2,
)
from copper_mas.contracts.runtime import WorkflowPlanV1
from copper_mas.data.leakage import FutureInformationError


def _request() -> ForecastRequestV2:
    decision = datetime(2025, 1, 1, 8)
    admission = AsOfAdmissionCardV2(
        origin_event_id="evt-1",
        decision_at=decision,
        feature_cutoff_at=decision,
        admission_status="ADMITTED",
    )
    observations = ObservationCardV2(
        origin_event_id="evt-1",
        decision_at=decision,
        observations=(
            ObservationItemV2(
                canonical_tag="origin_cu_g_l",
                value=5.0,
                unit="g/L",
                event_at=decision,
                available_at=decision,
                matched_anchor_at=decision,
                age_hours=0,
                missing=False,
            ),
        ),
    )
    return ForecastRequestV2(request_id="req-1", admission=admission, observations=observations)


def test_frozen_plan_rejects_reordered_agents() -> None:
    with pytest.raises(ValueError, match="节点顺序"):
        WorkflowPlanV1(
            run_id="run-1",
            ordered_nodes=("A1", "A4", "A2", "A5"),
            numeric_predictor_ref="model-1",
            external_test_access="DENIED",
        )


def test_offline_graph_has_zero_external_calls_and_audit_passes() -> None:
    result = run_offline_graph(
        plan=fixed_offline_plan(run_id="run-1", numeric_predictor_ref="model-1"),
        request=_request(),
        feature_row={
            "origin_cu_g_l": 5.0,
            "origin_as_mg_l": 1000.0,
            "phase1_stage12_current_ka__t_minus_0h": 1.0,
            "phase2_stage12_current_ka__t_minus_0h": 0.0,
            "stage3_current_a__t_minus_0h": 1.0,
            "stage4_current_a__t_minus_0h": 0.0,
        },
        predict=lambda row, mode: NumericPrediction(
            predicted_cu_g_l=float(row["origin_cu_g_l"]),
            predicted_as_mg_l=float(row["origin_as_mg_l"]),
        ),
    )
    assert result.audit.passed
    assert result.resources.total_calls == 0
    assert result.resources.total_input_tokens == 0
    assert len(result.resources.records) == 4


def test_graph_rejects_target_in_feature_row() -> None:
    with pytest.raises(FutureInformationError):
        run_offline_graph(
            plan=fixed_offline_plan(run_id="run-2", numeric_predictor_ref="model-1"),
            request=_request(),
            feature_row={"target_cu_g_l": 1.0},
            predict=lambda row, mode: NumericPrediction(
                predicted_cu_g_l=1.0,
                predicted_as_mg_l=1.0,
            ),
        )


def test_graph_rejects_invalid_prediction_interval() -> None:
    with pytest.raises(ValueError, match="点预测不在区间"):
        run_offline_graph(
            plan=fixed_offline_plan(run_id="run-3", numeric_predictor_ref="model-1"),
            request=_request(),
            feature_row={
                "phase1_stage12_current_ka__t_minus_0h": 1.0,
                "phase2_stage12_current_ka__t_minus_0h": 0.0,
                "stage3_current_a__t_minus_0h": 1.0,
                "stage4_current_a__t_minus_0h": 0.0,
            },
            predict=lambda row, mode: NumericPrediction(
                predicted_cu_g_l=5.0,
                predicted_as_mg_l=1000.0,
                cu_interval_low=6.0,
                cu_interval_high=7.0,
            ),
        )
