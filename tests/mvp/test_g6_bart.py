import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import norm

from copper_mvp.bart_trees import draw_predictions, mixture_quantiles
from copper_mvp.model_lifecycle import ModelLifecycle
from copper_mvp.model_registry import ComparisonRequest


def test_numeric_bart_preserves_split_equality_and_leaf_signs():
    tree = {
        "offsets": np.array([0, 3]),
        "features": np.array([0, -1, -1], dtype=np.int32),
        "values": np.array([0.5, -2.0, 4.0]),
        "left": np.array([1, -1, -1]),
        "right": np.array([2, -1, -1]),
        "draw_trees": np.array([[0], [0]], dtype=np.int32),
    }
    actual = draw_predictions(tree, np.array([[0.5], [np.nextafter(0.5, 1.0)]]))
    np.testing.assert_array_equal(actual, [[-2.0, 4.0], [-2.0, 4.0]])


def test_posterior_quantiles_match_normal_and_scale_transformation():
    means = np.full((20, 2), [2.0, 4.0])
    quantiles = mixture_quantiles(means, np.full(20, 0.3))
    expected = np.array([2.0, 4.0])[:, None] + 0.3 * norm.ppf([0.1, 0.5, 0.9])
    np.testing.assert_allclose(quantiles, expected, atol=1e-10)
    np.testing.assert_allclose(mixture_quantiles(1000 * means, np.full(20, 300.0)), 1000 * quantiles, atol=1e-7)


def test_long_budget_is_explicitly_limited_to_bart_comparisons():
    with pytest.raises(ValidationError):
        ComparisonRequest(request_key="legacy", max_wall_seconds=1801)
    request = ComparisonRequest(request_key="bart", methods=("Persistence", "BART"), max_wall_seconds=21600)
    assert request.max_wall_seconds == 21600


def test_unconverged_sampling_blocks_model_release():
    metric = {
        "n": 100,
        "coverage": 1.0,
        "negative_predictions": 0,
        "p95_ms": 1.0,
        "mae": 0.4,
        "baseline_mae": 1.0,
        "paired_ci95": {"high": -0.2},
        "sampling_diagnostics_passed": False,
    }
    gate = ModelLifecycle.quality_gate({"benchmark": metric}, None, "forecast")
    assert not gate["passed"] and "SAMPLING_DIAGNOSTICS" in gate["failures"]
