import numpy as np
import pytest

from copper_mvp.tabular_models import TabularModel
from copper_mvp.tabular_registry import TRAINING


@pytest.mark.parametrize("method", ["TabNet", "FTTransformer", "NODE"])
def test_mlp_control_matches_parameter_and_optimizer_budget(method):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(65, 114))
    X[:, 110:] = rng.integers(0, 3, (65, 4))
    y = X[:, :2] + rng.normal(scale=0.1, size=(65, 2))
    settings = {**TRAINING, "epochs": 1, "batch_size": 32}
    control = TabularModel(method, 17, settings, reference=True).fit(X, y)
    info = control.fit_metadata
    assert info["role"] == "parameter_matched_mlp_control"
    assert info["parameter_budget_relative_difference"] < 0.001
    assert info["optimizer_updates"] == 2 and info["epochs"] == 1
    assert np.isfinite(control.predict(X[:3])).all()
