from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from copper_mvp.api import create_app
from copper_mvp.common import WorkbenchError, digest, file_hash, write_json
from copper_mvp.data_contracts import LabelRecord
from copper_mvp.labels import LabelLedger
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_evaluation import evaluate_comparison, paired_week_interval, read_training
from copper_mvp.model_registry import ComparisonPredictionRequest, ComparisonRequest, METHOD_IDS, catalog
from copper_mvp.model_training import load_registered_model, train_comparison
from copper_mvp.modeling import make_model


def arrays(n=101):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(n, 114))
    X[:, :2] += 10
    X[:, 110:] = rng.integers(0, 3, size=(n, 4))
    X[::7, 4] = np.nan
    X[:, 7] = np.nan
    y = X[:, :2] + np.column_stack((0.2 * X[:, 2], -0.4 * X[:, 3]))
    return X, y


class SyntheticData:
    def __init__(self, root, perturb=False):
        root.mkdir(parents=True, exist_ok=True)
        X, y = arrays()
        if perturb:
            y[40:60] += 100
        self.ids = [f"synthetic-{i:03}" for i in range(101)]
        times = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(101)]
        self.frame = pd.DataFrame({"origin_event_id": self.ids, "decision_at": times}, index=self.ids)
        self.feature_columns = [f"input_{i}" for i in range(114)]
        self.X = pd.DataFrame(X, index=self.ids, columns=self.feature_columns)
        self.primary_ids = set(self.ids[:100])
        self.dataset_version = "synthetic-development-v1"
        rows = []
        self.fold_for = {}
        for fold, train_n, valid_start in (("FOLD_1", 40, 40), ("FOLD_2", 70, 70)):
            for i in list(range(train_n)) + list(range(valid_start, valid_start + 20)):
                role = "TRAIN" if i < train_n else "VALIDATION"
                rows.append({"fold_id": fold, "fold_role": role, "origin_event_id": self.ids[i],
                             "fold_fit_cutoff_at": times[valid_start].isoformat()})
                if role == "VALIDATION":
                    self.fold_for[self.ids[i]] = fold
        self.cv = pd.DataFrame(rows)
        records = []
        for i in range(100):
            for j, (target, unit) in enumerate((("cu", "g/L"), ("as", "mg/L"))):
                records.append(LabelRecord(event_id=self.ids[i], pair_id="pair-" + self.ids[i], target=target,
                    value=float(y[i, j]), unit=unit, decision_at=times[i], available_at=times[i + 1],
                    assumed_sample_time=times[i + 1] - timedelta(hours=2), quality_eligible=True,
                    source_kind="synthetic_fixture", source_version="synthetic-labels-v1"))
        self.ledger = LabelLedger(records)
        self.source_paths = {}
        for name in ("features", "admissions", "index", "folds", "feature_contract", "timing_contract", "labels", "pairing"):
            path = root / (name + ".json")
            path.write_text(json.dumps({"kind": "synthetic", "name": name, "perturbed": perturb if name == "labels" else False}), encoding="utf-8")
            self.source_paths[name] = path
        self.paths = SimpleNamespace(sources=lambda: self.source_paths)

    def row(self, event):
        if event not in self.frame.index:
            raise WorkbenchError("没有该合成事件", "EVENT_NOT_FOUND")
        return self.frame.loc[event]

    def training_ids(self, fold):
        if fold is None:
            return self.ids[:100]
        return self.cv.loc[self.cv.fold_id.eq(fold) & self.cv.fold_role.eq("TRAIN"), "origin_event_id"].tolist()


class SyntheticService:
    def __init__(self, data, *args):
        self.labels = data.ledger
    def snapshot(self, event):
        return {"id": "synthetic-snapshot:" + event}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("g2a")
    data = SyntheticData(root / "inputs")
    with ExitStack() as stack:
        for module in ("model_training", "model_evaluation", "model_comparisons"):
            stack.enter_context(patch("copper_mvp." + module + ".DataService", SyntheticService))
        output = root / ("a" * 32)
        request = ComparisonRequest(request_key="synthetic-full")
        manifest = train_comparison(data, request, output)
        result = evaluate_comparison(data, output)
        write_json(output / "state.json", {"run_id": output.name, "status": "completed", "created_at": "synthetic",
                   "fingerprint": "synthetic", "request": request.model_dump(mode="json"),
                   "evaluation_sha256": file_hash(output / "evaluation.json")})
        yield data, output, manifest, result


def test_registry_and_request_contracts():
    items = catalog()["items"]
    assert [m["method_id"] for m in items] == list(METHOD_IDS)
    assert sum(m["requires_fit"] for m in items) == 5
    assert all(not m["automatic_promotion"] and not m["causal_control"] for m in items)
    for payload in ({"methods": ("ElasticNet",)}, {"methods": ("Persistence", "Persistence")},
                    {"methods": ("Persistence", "unknown")}, {"target_cu_g_l": 1}):
        with pytest.raises(ValidationError):
            ComparisonRequest(request_key="test", **payload)


@pytest.mark.parametrize("method", METHOD_IDS)
def test_adapters_fit_roundtrip_and_missing_values(method, tmp_path):
    X, y = arrays()
    model = RegisteredModel(method).fit(X[:80], y[:80])
    predicted = model.predict(X[80:])
    assert predicted.shape == (21, 2) and np.isfinite(predicted).all()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(X[80:]), predicted)
    if method == "Persistence":
        np.testing.assert_array_equal(predicted, X[80:, :2])


@pytest.mark.parametrize("method", ("DeltaRidge", "DeltaHGB"))
def test_existing_method_numeric_parity(method):
    X, y = arrays()
    actual = RegisteredModel(method).fit(X[:80], y[:80]).predict(X[80:])
    expected = np.column_stack([X[80:, t] + make_model(method).fit(X[:80], y[:80, t] - X[:80, t]).predict(X[80:]) for t in range(2)])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ("ElasticNet", "Huber", "PLS"))
def test_target_unit_rescaling_and_training_only_transform(method):
    # Positive residual noise and an overdetermined fit avoid the zero-scale Huber degeneracy.
    X, y = arrays(260)
    y += np.random.default_rng(23).normal(0, 0.03, y.shape)
    model = RegisteredModel(method).fit(X[:200], y[:200])
    scaled_X, scaled_y = X.copy(), y.copy()
    scaled_X[:, 1] *= 1000
    scaled_y[:, 1] *= 1000
    second = RegisteredModel(method).fit(scaled_X[:200], scaled_y[:200])
    predicted = model.predict(X[200:])
    transformed = second.predict(scaled_X[200:])
    transformed[:, 1] /= 1000
    np.testing.assert_allclose(transformed, predicted, rtol=1e-4, atol=1e-4)
    fitted = model.models[0].regressor_.named_steps["preprocess"].named_transformers_["numeric"]
    before = fitted.named_steps["scale"].mean_.copy()
    shifted = X[200:].copy(); shifted[:, 2:110] *= 1000
    model.predict(shifted)
    np.testing.assert_array_equal(fitted.named_steps["scale"].mean_, before)
    expected_delta = y[:200] - X[:200, :2]
    mean = model.models[0].transformer_.mean_
    np.testing.assert_allclose(mean, expected_delta.mean(axis=0) if method == "PLS" else [expected_delta[:, 0].mean()])


def test_complete_cohort_artifacts_and_independent_metrics(experiment):
    data, root, manifest, result = experiment
    assert result["status"] == "completed" and result["common_events"] == 40
    assert len(result["metrics"]) == 12 and len(result["fold_metrics"]) == 24
    assert len(manifest["artifacts"]) == 18
    assert sum(a["requires_fit"] for a in manifest["artifacts"]) == 15
    assert all(v["coverage"] == 1 for v in result["coverage"].values())
    records = data.ledger.latest(data.frame.decision_at.max())
    events = sorted(data.fold_for)
    expected = np.mean([abs(data.X.loc[e].iloc[0] - records[(e, "cu")].value) for e in events])
    persistence = next(m for m in result["metrics"] if m["method_id"] == "Persistence" and m["target"] == "cu")
    assert persistence["mae"] == pytest.approx(expected)
    assert not result["automatic_promotion"] and not result["optimization_proxy_approval"]
    for method in METHOD_IDS:
        prediction = ComparisonService(root.parent, data).predict(root.name, ComparisonPredictionRequest(event_id=events[0], method_id=method))
        rows = pd.read_csv(root / "oof_predictions.csv")
        saved = rows[rows.event_id.eq(events[0]) & rows.method_id.eq(method)].iloc[0]
        assert prediction["predictions"]["cu"]["value"] == pytest.approx(saved.cu)
        assert prediction["predictions"]["as"]["value"] == pytest.approx(saved["as"])


def test_validation_label_perturbation_does_not_change_first_fold_fit(experiment, tmp_path):
    data, original, _, _ = experiment
    altered = SyntheticData(tmp_path / "changed-inputs", perturb=True)
    output = tmp_path / "comparison"
    train_comparison(altered, ComparisonRequest(request_key="perturb", methods=("Persistence", "ElasticNet")), output)
    left_manifest, left_protocol = read_training(original)
    right_manifest, right_protocol = read_training(output)
    left, _ = load_registered_model(original, left_manifest, left_protocol, "ElasticNet", "FOLD_1")
    right, _ = load_registered_model(output, right_manifest, right_protocol, "ElasticNet", "FOLD_1")
    values = data.X.loc[data.ids[40:60]].to_numpy()
    np.testing.assert_array_equal(left.predict(values), right.predict(values))


def test_tampered_model_and_prediction_files_are_rejected(experiment, tmp_path):
    data, root, _, _ = experiment
    copied = tmp_path / ("b" * 32)
    shutil.copytree(root, copied)
    manifest, protocol = read_training(copied)
    entry = next(a for a in manifest["artifacts"] if a["method_id"] == "ElasticNet")
    path = copied / entry["path"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(WorkbenchError, match="哈希"):
        load_registered_model(copied, manifest, protocol, "ElasticNet", entry["fold_id"])
    with (copied / "oof_predictions.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(WorkbenchError, match="校验"):
        read_training(copied)


def test_common_cohort_excludes_failures_without_hiding_missing_rows(experiment, tmp_path):
    data, root, _, _ = experiment
    copied = tmp_path / "failure"
    shutil.copytree(root, copied, ignore=shutil.ignore_patterns("evaluation.json", "metrics.csv", "fold_metrics.csv", "report.md"))
    rows = pd.read_csv(copied / "oof_predictions.csv")
    failed_event = rows.event_id.iloc[0]
    mask = rows.event_id.eq(failed_event) & rows.method_id.eq("Huber")
    rows.loc[mask, ["cu", "as"]] = np.nan
    rows.loc[mask, "status"] = "failed"
    rows.to_csv(copied / "oof_predictions.csv", index=False)
    manifest = json.loads((copied / "training_manifest.json").read_text())
    manifest["oof_sha256"] = file_hash(copied / "oof_predictions.csv")
    write_json(copied / "training_manifest.json", manifest)
    result = evaluate_comparison(data, copied)
    assert result["common_events"] == 39 and result["status"] == "partial_failure"
    assert result["coverage"]["Huber"]["missing_or_failed_events"] == 1
    assert all(row["n"] == 39 for row in result["metrics"])


def test_week_block_interval_is_paired_and_handles_small_support():
    settings = {"minimum_blocks": 20, "replicates": 1000, "seed": 17}
    dates = pd.date_range("2024-01-01", periods=30, freq="7D", tz="UTC")
    interval = paired_week_interval(np.ones(30), dates, settings)
    assert interval["low"] == 1 and interval["high"] == 1
    assert paired_week_interval(np.ones(2), dates[:2], settings)["reason"] == "INSUFFICIENT_TIME_BLOCKS"


def test_api_idempotency_future_field_rejection_and_legacy_separation(experiment, tmp_path):
    data, _, _, _ = experiment
    with TestClient(create_app(tmp_path, data)) as client:
        assert len(client.get("/api/v2/models").json()["items"]) == 6
        request = {"request_key": "api-g2", "methods": ["Persistence", "ElasticNet"]}
        created = client.post("/api/v2/model-comparisons", json=request)
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        for _ in range(200):
            state = client.get("/api/v2/model-comparisons/" + run_id).json()
            if state["status"] == "completed":
                break
            time.sleep(0.05)
        assert state["status"] == "completed" and state["result"]["common_events"] == 40
        assert client.post("/api/v2/model-comparisons", json=request).json()["reused"]
        assert client.post("/api/v2/model-comparisons", json={**request, "seed": 1}).status_code == 409
        assert client.post("/api/v2/model-comparisons", json={**request, "target_cu_g_l": 1}).status_code == 422
        assert client.get("/api/models").json()["items"] == []
        assert client.get("/api/v2/model-comparisons/" + run_id + "/export?format=csv").status_code == 200


def test_budget_boundary_preserves_completed_artifacts(experiment, tmp_path, monkeypatch):
    data, _, _, _ = experiment
    import copper_mvp.model_training as training
    actual_clock = time.perf_counter
    extra = [0.0]
    monkeypatch.setattr(training.time, "perf_counter", lambda: actual_clock() + extra[0])
    def progress(value):
        if value["phase"] == "trained":
            extra[0] = 1000.0
    output = tmp_path / "budget"
    with pytest.raises(WorkbenchError, match="时间预算"):
        train_comparison(data, ComparisonRequest(request_key="budget", methods=("Persistence", "ElasticNet"), max_wall_seconds=30), output, progress)
    manifest = json.loads((output / "training_manifest.json").read_text())
    assert manifest["status"] == "time_budget_exceeded"
    assert len(manifest["artifacts"]) == 1 and manifest["artifacts"][0]["status"] == "completed"
    assert len(pd.read_csv(output / "oof_predictions.csv")) == 20


def test_evaluation_binding_rejects_changed_training_manifest(experiment, tmp_path):
    data, root, _, _ = experiment
    copied = tmp_path / ("c" * 32)
    shutil.copytree(root, copied)
    path = copied / "training_manifest.json"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    service = ComparisonService(tmp_path, data)
    with pytest.raises(WorkbenchError, match="绑定失效"):
        service.get(copied.name)
    with pytest.raises(WorkbenchError, match="绑定失效"):
        service.predict(copied.name, ComparisonPredictionRequest(event_id=data.ids[40], method_id="ElasticNet"))
