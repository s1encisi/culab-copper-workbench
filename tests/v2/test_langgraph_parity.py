from __future__ import annotations

from datetime import datetime

import pytest

from copper_langgraph_v2.bridge import (
    AsOfAdmissionCardV2,
    ForecastRequestV2,
    NumericPrediction,
    ObservationCardV2,
    ObservationItemV2,
    fixed_offline_plan,
    run_offline_graph,
)
from copper_langgraph_v2.graph import (
    LangGraphRunner,
    RunIdConflictError,
    persistence_predictor,
)


def _request(*, request_id: str = "req-v2") -> ForecastRequestV2:
    decision = datetime(2025, 1, 1, 8)
    admission = AsOfAdmissionCardV2(
        origin_event_id="evt-v2",
        decision_at=decision,
        feature_cutoff_at=decision,
        admission_status="ADMITTED",
    )
    observations = ObservationCardV2(
        origin_event_id="evt-v2",
        decision_at=decision,
        observations=(
            ObservationItemV2(
                canonical_tag="origin_cu_g_l",
                value=5.25,
                unit="g/L",
                event_at=decision,
                available_at=decision,
                matched_anchor_at=decision,
                age_hours=0,
                missing=False,
            ),
        ),
    )
    return ForecastRequestV2(request_id=request_id, admission=admission, observations=observations)


def _features() -> dict[str, float]:
    return {
        "origin_cu_g_l": 5.25,
        "origin_as_mg_l": 1200.0,
        "phase1_stage12_current_ka__t_minus_0h": 1.0,
        "phase2_stage12_current_ka__t_minus_0h": 0.0,
        "stage3_current_a__t_minus_0h": 1.0,
        "stage4_current_a__t_minus_0h": 0.0,
    }


def test_langgraph_matches_v1_semantics(tmp_path) -> None:
    plan = fixed_offline_plan(run_id="parity-1", numeric_predictor_ref="PERSISTENCE_CURRENT_RESULT_V1")
    request = _request()
    features = _features()
    expected = run_offline_graph(
        plan=plan,
        request=request,
        feature_row=features,
        predict=persistence_predictor,
    )
    with LangGraphRunner(
        checkpoint_path=tmp_path / "checkpoints.sqlite",
        predict=persistence_predictor,
    ) as runner:
        actual = runner.invoke(plan=plan, request=request, feature_row=features)

    assert actual["trace"] == ["A1", "A2", "A4", "A5"]
    assert actual["process_mode"] == expected.process_mode.model_dump(mode="json")
    assert actual["prediction"] == expected.prediction.model_dump(mode="json")
    assert actual["audit"] == expected.audit.model_dump(mode="json")
    assert actual["resources"]["total_calls"] == 0
    assert actual["resources"]["total_input_tokens"] == 0
    assert actual["resources"]["total_output_tokens"] == 0
    assert actual["resources"]["total_estimated_cost_cny"] == 0


def test_future_target_is_fail_closed(tmp_path) -> None:
    plan = fixed_offline_plan(run_id="fail-a1", numeric_predictor_ref="model-v1")
    features = {**_features(), "target_cu_g_l": 6.0}
    with LangGraphRunner(checkpoint_path=tmp_path / "fail.sqlite", predict=persistence_predictor) as runner:
        result = runner.invoke(plan=plan, request=_request(), feature_row=features)

    assert result["completed"] is True
    assert result["blocked"] is True
    assert result["prediction"] is None
    assert result["audit"]["passed"] is False
    assert result["trace"] == ["A1", "FAIL_CLOSED"]


def test_invalid_numeric_output_is_fail_closed(tmp_path) -> None:
    def invalid_predictor(row, mode):
        del row, mode
        return NumericPrediction(
            predicted_cu_g_l=5.0,
            predicted_as_mg_l=1000.0,
            cu_interval_low=6.0,
            cu_interval_high=7.0,
        )

    plan = fixed_offline_plan(run_id="fail-a4", numeric_predictor_ref="bad-model")
    with LangGraphRunner(checkpoint_path=tmp_path / "invalid.sqlite", predict=invalid_predictor) as runner:
        result = runner.invoke(plan=plan, request=_request(), feature_row=_features())

    assert result["blocked"] is True
    assert result["prediction"] is None
    assert result["failure"]["node"] == "A4"
    assert result["trace"] == ["A1", "A2", "A4", "FAIL_CLOSED"]


def test_run_id_is_idempotent_and_conflict_safe(tmp_path) -> None:
    plan = fixed_offline_plan(run_id="idempotent-1", numeric_predictor_ref="PERSISTENCE_CURRENT_RESULT_V1")
    database = tmp_path / "idempotent.sqlite"
    with LangGraphRunner(checkpoint_path=database, predict=persistence_predictor) as runner:
        first = runner.invoke(plan=plan, request=_request(), feature_row=_features())
        history_before = runner.history_count(plan.run_id)
        replay = runner.invoke(plan=plan, request=_request(), feature_row=_features())
        history_after = runner.history_count(plan.run_id)
        with pytest.raises(RunIdConflictError):
            runner.invoke(
                plan=plan,
                request=_request(request_id="different-request"),
                feature_row=_features(),
            )

    assert database.is_file()
    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert history_before == history_after
