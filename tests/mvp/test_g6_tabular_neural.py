import joblib
import numpy as np
import pytest

from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.tabular_inputs import NeuralInputs
from copper_mvp.tabular_registry import TRAINING
from copper_mvp.tabular_runtime import tabular_runtime


def test_missing_mask_and_scaling_use_training_state():
    X = np.zeros((6, 114))
    X[:, 0] = np.arange(6)
    X[:, 1] = 10
    X[:, 110:] = 1
    X[0, 3] = np.nan
    pre = NeuralInputs().fit(X)
    means = pre.scaler.mean_.copy()
    query = X[:2].copy()
    query[1, 4] = np.nan
    query[0, 0] = 10000
    numerical, categories = pre.transform(query)
    np.testing.assert_array_equal(numerical[:, 110:], np.isnan(query[:, :110]))
    np.testing.assert_array_equal(means, pre.scaler.mean_)
    assert numerical.shape == (2, 220) and categories.shape == (2, 4)


@pytest.mark.parametrize("method", ["TabNet", "FTTransformer", "NODE"])
def test_registered_network_trains_and_roundtrips(method, tmp_path, monkeypatch):
    monkeypatch.setitem(TRAINING, "epochs", 2)
    monkeypatch.setitem(TRAINING, "batch_size", 32)
    torch = tabular_runtime()
    rng = np.random.default_rng(21)
    X = rng.normal(size=(65, 114))
    X[:, 0] = 4 + 0.2 * X[:, 0]
    X[:, 1] = 300 + 10 * X[:, 1]
    X[:, 110:] = rng.integers(0, 3, (65, 4))
    X[::7, 3] = np.nan
    y = X[:, :2] + np.column_stack((0.2 * X[:, 2], 20 * X[:, 4]))
    before = torch.get_rng_state().clone()
    model = RegisteredModel(method, 21).fit(X, y, sample_weight=np.linspace(0.5, 1.5, len(X)))
    assert torch.equal(before, torch.get_rng_state())
    values = model.predict(X[:5])
    path = tmp_path / "network.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(X[:5]), values, rtol=1e-6, atol=1e-6)
    assert torch.equal(before, torch.get_rng_state())
    changes = model.fit_metadata["parameter_max_changes"]
    assert max(changes.values()) > 0 and np.isfinite(values).all()
    assert model.fit_metadata["optimizer_updates"] == 4
    if method == "NODE":
        for name in ("selection", "thresholds", "responses"):
            assert any(change > 0 for key, change in changes.items() if name in key)
