import numpy as np
import pytest
from scipy.stats import norm
from pydantic import ValidationError
from copper_mvp.bart_trees import draw_predictions, mixture_quantiles
from copper_mvp.model_registry import ComparisonRequest
from copper_mvp.model_lifecycle import ModelLifecycle


def test_numeric_bart_preserves_split_equality_and_leaf_signs():
    tree = {"offsets": np.array([0, 3]), "features": np.array([0, -1, -1], dtype=np.int32),
            "values": np.array([.5, -2., 4.]), "left": np.array([1, -1, -1]),
            "right": np.array([2, -1, -1]), "draw_trees": np.array([[0], [0]], dtype=np.int32)}
    actual = draw_predictions(tree, np.array([[.5], [np.nextafter(.5, 1.)]]))
    np.testing.assert_array_equal(actual, [[-2., 4.], [-2., 4.]])


def test_posterior_quantiles_match_normal_and_scale_transformation():
    means = np.full((20, 2), [2., 4.])
    quantiles = mixture_quantiles(means, np.full(20, .3))
    expected = np.array([2., 4.])[:, None]+.3*norm.ppf([.1, .5, .9])
    np.testing.assert_allclose(quantiles, expected, atol=1e-10)
    np.testing.assert_allclose(mixture_quantiles(1000*means, np.full(20, 300.)),
                               1000*quantiles, atol=1e-7)


def test_long_budget_is_explicitly_limited_to_bart_comparisons():
    with pytest.raises(ValidationError):
        ComparisonRequest(request_key="legacy", max_wall_seconds=1801)
    request = ComparisonRequest(request_key="bart", methods=("Persistence", "BART"), max_wall_seconds=21600)
    assert request.max_wall_seconds == 21600


def test_unconverged_sampling_blocks_model_release():
    metric = {"n": 100, "coverage": 1., "negative_predictions": 0, "p95_ms": 1.,
              "mae": .4, "baseline_mae": 1., "paired_ci95": {"high": -.2},
              "sampling_diagnostics_passed": False}
    gate = ModelLifecycle.quality_gate({"benchmark": metric}, None, "forecast")
    assert not gate["passed"] and "SAMPLING_DIAGNOSTICS" in gate["failures"]
