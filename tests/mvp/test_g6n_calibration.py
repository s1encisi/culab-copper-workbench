from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from copper_mvp.calibration import SplitC90, finite_sample_quantile, interval_metrics
from copper_mvp.calibration_training import temporal_split
from copper_mvp.common import WorkbenchError
from copper_mvp.data_contracts import LabelRecord, source_time


def test_finite_sample_quantile_uses_order_statistic_without_interpolation():
    assert finite_sample_quantile(np.arange(9), 0.9) == 8
    with pytest.raises(WorkbenchError) as error:
        finite_sample_quantile(np.arange(8), 0.9)
    assert error.value.code == "INSUFFICIENT_CALIBRATION"


def test_marginal_and_joint_coverage_are_distinct_and_unit_equivariant():
    point = np.zeros((100, 2))
    y = np.column_stack([np.arange(100), np.arange(100)[::-1] * 10])
    model = SplitC90.fit(point, y, [1, 10])
    intervals = model.intervals(point)
    marginal = intervals["marginal_c90"]
    joint = intervals["joint_c90"]
    marginal_joint = ((y >= marginal[:, 0, :]) & (y <= marginal[:, 1, :])).all(axis=1).mean()
    joint_joint = ((y >= joint[:, 0, :]) & (y <= joint[:, 1, :])).all(axis=1).mean()
    assert marginal_joint == 0.82
    assert joint_joint >= 0.9
    multiplier = np.array([1000.0, 0.001])
    transformed = SplitC90.fit(point * multiplier, y * multiplier, np.array([1, 10]) * multiplier)
    np.testing.assert_allclose(transformed.intervals(point * multiplier)["joint_c90"], joint * multiplier)
    assert not model.manifest()["marginal_is_joint"]


def test_zero_training_scale_does_not_fabricate_joint_normalization():
    model = SplitC90.fit(np.zeros((60, 2)), np.ones((60, 2)), [0, 2])
    assert model.joint_radius is None
    assert model.manifest()["joint_unavailable_reason"] == "ZERO_TRAINING_SCALE"
    assert "joint_c90" not in model.intervals(np.zeros((2, 2)))


def test_interval_scoring_and_invalid_bounds():
    result = interval_metrics([3, 0], [2, 2], [1, 1], [4, 4], 0.9, scale=2)
    assert result["picp"] == 0.5
    assert result["mean_width"] == 3 and result["normalized_width"] == 1.5
    assert result["wis"] == pytest.approx((0.65 / 1.5 + 2.15 / 1.5) / 2)
    with pytest.raises(WorkbenchError):
        interval_metrics([1], [1], [2], [0])


def test_calibration_split_purges_labels_not_available_at_base_fit_time():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    identifiers = [f"event-{i:03}" for i in range(110)]
    rows = {event: SimpleNamespace(decision_at=start + timedelta(hours=i)) for i, event in enumerate(identifiers)}
    data = SimpleNamespace(row=lambda event: rows[event])
    records = []
    for i, event in enumerate(identifiers):
        for target, unit, delay in (("cu", "g/L", 1), ("as", "mg/L", 3)):
            available = start + timedelta(hours=i + delay)
            records.append(
                LabelRecord(
                    event_id=event,
                    pair_id=event,
                    target=target,
                    value=float(i + 1),
                    unit=unit,
                    decision_at=rows[event].decision_at,
                    available_at=available,
                    assumed_sample_time=available - timedelta(hours=2),
                    quality_eligible=True,
                    source_version="synthetic-v1",
                    source_kind="synthetic_fixture",
                )
            )

    class Ledger:
        def latest(self, cutoff):
            return {(r.event_id, r.target): r for r in records if r.available_at <= source_time(cutoff)}

    request = SimpleNamespace(minimum_calibration=10, minimum_fit=20, calibration_fraction=0.2)
    split = temporal_split(
        data, Ledger(), identifiers[:97], identifiers[100:], start + timedelta(hours=100), request, "fold"
    )
    assert split["base_fit_cutoff_at"] == (start + timedelta(hours=77)).isoformat()
    assert split["purged_ids"] == identifiers[75:77]
    assert split["fit_ids"] == identifiers[:75]
    assert split["calibration_ids"] == identifiers[77:97]
    assert not set(split["validation_ids"]) & (set(split["fit_ids"]) | set(split["calibration_ids"]))
