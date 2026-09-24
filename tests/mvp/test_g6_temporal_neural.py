from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from copper_mvp.common import WorkbenchError
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.neural_runtime import load_tensor_runtime
from copper_mvp.temporal_inputs import OFFSETS, TemporalInputs, raw_anchor_windows
from copper_mvp.temporal_registry import TEMPORAL_ARCHITECTURES, TEMPORAL_TRAINING


def sample_data():
    rng = np.random.default_rng(32)
    ids = [f"e{i}" for i in range(40)]
    tags = [f"signal{i}" for i in range(27)]
    records = {}
    for offset in OFFSETS:
        for tag in tags:
            records[tag + f"__t_minus_{offset}h"] = rng.normal(size=40)
            records[tag + f"__t_minus_{offset}h__age_h"] = np.ones(40)
    records["decision_at"] = pd.date_range("2024-01-01", periods=40, tz="UTC")
    frame = pd.DataFrame(records, index=ids)
    X = np.zeros((40, 114))
    X[:, 0] = 4 + 0.2 * rng.normal(size=40)
    X[:, 1] = 300 + 10 * rng.normal(size=40)
    X[:, 110:] = rng.integers(0, 3, (40, 4))
    matrix = pd.DataFrame(X, index=ids)
    data = SimpleNamespace(frame=frame, X=matrix, tags=tags, row=lambda event: frame.loc[event])
    labels = {
        (e, t): SimpleNamespace(
            quality_eligible=True,
            value=float(
                matrix.loc[e, 0 if t == "cu" else 1] + (0.2 if t == "cu" else 20) * frame.loc[e, "signal0__t_minus_0h"]
            ),
        )
        for e in ids
        for t in ("cu", "as")
    }
    return data, ids, labels


@pytest.mark.parametrize("method", ["GRU", "LSTM", "CausalTCN"])
def test_future_sequence_values_do_not_change_past_outputs(method):
    torch = load_tensor_runtime()
    from copper_mvp.temporal_networks import TemporalNetwork

    torch.manual_seed(5)
    network = TemporalNetwork(method, TEMPORAL_ARCHITECTURES[method]).eval()
    X = torch.randn(2, 7, 109)
    changed = X.clone()
    changed[:, 4:] += 100
    with torch.no_grad():
        a = network.encode(X)
        b = network.encode(changed)
    torch.testing.assert_close(a[:, :4], b[:, :4], rtol=1e-6, atol=1e-7)


def test_anchor_times_and_training_preprocessing_remain_causal():
    data, ids, _ = sample_data()
    pre = TemporalInputs().fit(data, ids[:24])
    original = pre.transform(data, ids[:3])[0]
    data.frame.loc[ids[-1], "signal0__t_minus_0h"] = 1e9
    np.testing.assert_array_equal(pre.transform(data, ids[:3])[0], original)
    data.frame.loc[ids[0], "signal0__t_minus_0h__age_h"] = -0.1
    with pytest.raises(WorkbenchError, match="锚点"):
        raw_anchor_windows(data, ids[:1])


@pytest.mark.parametrize("method", ["GRU", "LSTM", "CausalTCN"])
def test_context_models_train_serialize_and_do_not_read_labels_at_prediction(method, tmp_path, monkeypatch):
    data, ids, labels = sample_data()
    import copper_mvp.temporal_models as module

    monkeypatch.setattr(
        module, "DataService", lambda data: SimpleNamespace(labels=SimpleNamespace(latest=lambda cutoff: labels))
    )
    monkeypatch.setitem(TEMPORAL_TRAINING, "epochs", 2)
    monkeypatch.setitem(TEMPORAL_TRAINING, "batch_size", 8)
    torch = load_tensor_runtime()
    before = torch.get_rng_state().clone()
    model = RegisteredModel(method, 42).fit_context(data, ids[:24], data.frame.loc[ids[24], "decision_at"])
    assert torch.equal(before, torch.get_rng_state())
    duplicate = RegisteredModel(method, 42).fit_context(data, ids[:24], data.frame.loc[ids[24], "decision_at"])
    np.testing.assert_allclose(
        duplicate.predict_context(data, ids[25:30]), model.predict_context(data, ids[25:30]), rtol=1e-6, atol=1e-6
    )
    assert torch.equal(before, torch.get_rng_state())

    def no_labels(*args):
        raise AssertionError("Prediction must not query target labels")

    monkeypatch.setattr(module, "DataService", no_labels)
    events = ids[25:30]
    prediction = model.predict_context(data, events)
    np.testing.assert_allclose(
        model.predict_context(data, list(reversed(events)))[::-1], prediction, rtol=1e-6, atol=1e-6
    )
    path = tmp_path / "sequence.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict_context(data, events), prediction, rtol=1e-6, atol=1e-6)
    assert np.isfinite(prediction).all()
    assert model.fit_metadata["hidden_state_carried_between_events"] is False
    assert max(model.fit_metadata["parameter_max_changes"].values()) > 0


def test_temporal_parameter_counts_are_comparable():
    torch = load_tensor_runtime()
    from copper_mvp.temporal_networks import TemporalNetwork

    counts = [
        sum(p.numel() for p in TemporalNetwork(m, TEMPORAL_ARCHITECTURES[m]).parameters())
        for m in ("GRU", "LSTM", "CausalTCN")
    ]
    assert max(counts) / min(counts) < 1.05
