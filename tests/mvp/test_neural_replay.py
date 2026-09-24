import numpy as np

from copper_mvp.neural_replay import neural_replay_error


def test_float32_replay_tolerance_is_not_prediction_quality_tolerance():
    origin = np.array([4.0, 6000.0])
    scale = np.array([0.4, 500.0])
    expected = np.array([4.7, 7000.0])
    okay, detail = neural_replay_error(expected + np.array([3e-8, 6.1e-5]), expected, origin, scale)
    assert okay and detail["maximum_tolerance_fraction"] < 1
    bad, _ = neural_replay_error(expected + np.array([0.01, 1.0]), expected, origin, scale)
    assert not bad


def test_tabpfn_has_a_separate_numerical_budget_and_rejects_material_drift():
    origin = np.array([4.0, 6000.0])
    scale = np.array([0.4, 500.0])
    expected = origin + np.array([0.1, 20.0])
    small = expected + scale * 2e-5
    assert not neural_replay_error(small, expected, origin, scale)[0]
    okay, details = neural_replay_error(small, expected, origin, scale, method="TabPFN")
    assert okay and details["absolute_tolerance"] == 1e-4
    assert not neural_replay_error(expected + scale * 0.001, expected, origin, scale, method="TabPFN")[0]
