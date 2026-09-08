from datetime import datetime

import pytest

from copper_mas.agents.mode import infer_process_mode
from copper_mas.data.leakage import FutureInformationError


def test_infer_process_mode_uses_latest_available_state() -> None:
    row = {
        "phase1_stage12_current_ka__t_minus_0h": 10,
        "phase1_stage12_voltage_v__t_minus_0h": 100,
        "phase1_stage12_active_cells__t_minus_0h": 8,
        "phase2_stage12_current_ka__t_minus_0h": 0,
        "phase2_stage12_voltage_v__t_minus_0h": 0,
        "phase2_stage12_active_cells__t_minus_0h": 0,
        "stage3_current_a__t_minus_0h": None,
        "stage3_current_a__t_minus_0h__missing": True,
        "stage3_current_a__t_minus_2h": 500,
        "stage3_voltage_v__t_minus_2h": 20,
        "stage3_flow_m3_h__t_minus_2h": 3,
        "stage4_current_a__t_minus_0h": 0,
        "stage4_voltage_v__t_minus_0h": 0,
        "stage4_flow_m3_h__t_minus_0h": 0,
    }

    card = infer_process_mode(
        origin_event_id="evt-1",
        decision_at=datetime(2025, 1, 1, 8),
        feature_row=row,
    )

    assert card.mode_code == (
        "P1_STAGE12_ON__P2_STAGE12_OFF__S3_ON__S4_OFF"
    )
    assert "stage3_current_a__t_minus_2h" in card.evidence_observation_ids
    assert card.confidence == 1.0


def test_unknown_and_signal_conflict_are_audited() -> None:
    row = {
        "phase1_stage12_current_ka__t_minus_0h": 0,
        "phase1_stage12_voltage_v__t_minus_0h": 1,
        "phase2_stage12_current_ka__t_minus_0h": None,
        "stage3_current_a__t_minus_0h": None,
        "stage4_current_a__t_minus_0h": None,
    }
    card = infer_process_mode(
        origin_event_id="evt-2",
        decision_at=datetime(2025, 1, 1, 8),
        feature_row=row,
    )
    assert "P1_STAGE12_SIGNAL_CONFLICT" in card.warnings
    assert "P2_STAGE12_CURRENT_MISSING" in card.warnings
    assert "S3_UNKNOWN" in card.mode_code


def test_future_keys_fail_closed() -> None:
    with pytest.raises(FutureInformationError):
        infer_process_mode(
            origin_event_id="evt-3",
            decision_at=datetime(2025, 1, 1, 8),
            feature_row={"target_cu_g_l": 1.0},
        )
