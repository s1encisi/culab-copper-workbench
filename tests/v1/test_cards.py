from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from copper_mas.contracts.cards import (
    AsOfAdmissionCardV2,
    ForecastRequestV2,
    ObservationCardV2,
    ObservationItemV2,
)


def make_cards():
    decision = datetime(2025, 6, 1, 8, 0)
    admission = AsOfAdmissionCardV2(
        origin_event_id="origin_1",
        decision_at=decision,
        feature_cutoff_at=decision,
        admission_status="ADMITTED",
        feature_group_counts={"stage34": 7},
    )
    observation = ObservationCardV2(
        origin_event_id="origin_1",
        decision_at=decision,
        observations=(
            ObservationItemV2(
                canonical_tag="stage3_current_a",
                value=1234.0,
                unit="A",
                event_at=decision - timedelta(hours=2),
                available_at=decision - timedelta(hours=2),
                matched_anchor_at=decision - timedelta(hours=2),
                age_hours=0,
                missing=False,
                source_file="source.xlsx",
                source_sheet="电积脱铜生产记录",
                source_row=10,
                quality_code="PASS",
            ),
        ),
    )
    return decision, admission, observation


def test_forecast_request_accepts_as_of_cards():
    _, admission, observation = make_cards()
    request = ForecastRequestV2(request_id="req_1", admission=admission, observations=observation)
    assert request.admission.origin_event_id == "origin_1"


def test_admission_rejects_future_field_by_schema():
    decision, _, _ = make_cards()
    with pytest.raises(ValidationError):
        AsOfAdmissionCardV2(
            origin_event_id="origin_1",
            decision_at=decision,
            feature_cutoff_at=decision,
            admission_status="ADMITTED",
            target_event_id="future_1",
        )


def test_observation_rejects_future_availability():
    decision, _, _ = make_cards()
    with pytest.raises(ValidationError):
        ObservationCardV2(
            origin_event_id="origin_1",
            decision_at=decision,
            observations=(
                ObservationItemV2(
                    canonical_tag="stage4_flow_m3_h",
                    value=10.0,
                    available_at=decision + timedelta(minutes=1),
                    missing=False,
                ),
            ),
        )


def test_rejected_admission_cannot_be_forecast_request():
    decision, _, observation = make_cards()
    rejected = AsOfAdmissionCardV2(
        origin_event_id="origin_1",
        decision_at=decision,
        feature_cutoff_at=decision,
        admission_status="REJECTED",
    )
    with pytest.raises(ValidationError):
        ForecastRequestV2(request_id="req_1", admission=rejected, observations=observation)
