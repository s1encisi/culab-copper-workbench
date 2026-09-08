from datetime import datetime

import pytest

from copper_mas.data.leakage import (
    FutureInformationError,
    assert_no_future_information,
    find_forbidden_paths,
)


@pytest.mark.parametrize(
    "key",
    [
        "target_event_id",
        "target_recorded_at",
        "target_cu_g_l",
        "next_event_time",
        "lead_to_recorded_hours",
        "forward_prediction_flag",
    ],
)
def test_forbidden_future_keys_are_detected(key):
    paths = find_forbidden_paths({"safe": {key: "future"}})
    assert paths == [f"$.safe.{key}"]


def test_nested_future_availability_fails_closed():
    decision = datetime(2025, 1, 1, 10, 0)
    payload = {"observations": [{"available_at": datetime(2025, 1, 1, 10, 1)}]}
    with pytest.raises(FutureInformationError):
        assert_no_future_information(payload, decision_at=decision)


def test_safe_payload_passes():
    decision = datetime(2025, 1, 1, 10, 0)
    payload = {
        "origin_event_id": "origin_1",
        "observations": [{"available_at": datetime(2025, 1, 1, 8, 0), "value": 1.0}],
    }
    assert_no_future_information(payload, decision_at=decision)
