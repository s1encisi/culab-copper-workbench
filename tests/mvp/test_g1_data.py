from __future__ import annotations

import json
import shutil
import warnings
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from copper_mvp.api import create_app
from copper_mvp.common import DEFAULT_RUNS_DIR, WorkbenchError
from copper_mvp.data import DataRepository
from copper_mvp.data_contracts import LabelRecord, ReplayQuery, TaskSpec, source_time
from copper_mvp.data_service import DataService
from copper_mvp.labels import LabelLedger
from copper_mvp.modeling import ModelManager


def pair(event="synthetic-a", day=1):
    decision = datetime(2025, 1, day, tzinfo=UTC)
    return tuple(
        LabelRecord(
            event_id=event,
            pair_id="pair-" + event,
            target=target,
            value=value,
            unit=unit,
            decision_at=decision,
            available_at=decision + timedelta(days=1),
            assumed_sample_time=decision + timedelta(hours=22),
            quality_eligible=True,
            source_version="synthetic-source-v1",
            source_kind="synthetic_fixture",
        )
        for target, value, unit in (("cu", 2.0, "g/L"), ("as", 5.0, "mg/L"))
    )


@pytest.fixture(scope="module")
def repository():
    return DataRepository()


@pytest.fixture
def selected(repository):
    return repository.events(scope="dual", limit=1)["items"][0]["event_id"]


def test_naive_query_time_and_future_request_fields_are_rejected():
    with pytest.raises(ValidationError):
        ReplayQuery(as_of="2025-01-01T00:00:00")
    with pytest.raises(ValidationError):
        ReplayQuery(as_of="2025-01-01T00:00:00Z", target_cu_g_l=10)
    with pytest.raises(ValidationError):
        ReplayQuery(model_profile="auto")
    assert source_time("2025-01-01 08:00:00") == datetime(2025, 1, 1, tzinfo=UTC)


def test_task_units_and_input_schema_are_enforced(repository, selected):
    task = DataService(repository).task_spec(selected)
    assert task.observed_sample_time is None and task.sample_delay_is_assumption
    assert task.assumed_sample_time == task.decision_at - timedelta(hours=2)
    body = task.model_dump()
    with pytest.raises(ValidationError):
        TaskSpec(**{**body, "input_columns": (*body["input_columns"], "target_cu_g_l")})
    with pytest.raises(ValidationError):
        TaskSpec(**{**body, "targets": ({"name": "cu", "unit": "mg/L"}, {"name": "as", "unit": "mg/L"})})
    with pytest.raises(ValidationError):
        TaskSpec(**{**body, "feature_cutoff_at": task.decision_at + timedelta(seconds=1)})


def test_snapshot_never_loads_outcomes(repository, selected, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prediction snapshot must not load labels")

    monkeypatch.setattr(LabelLedger, "from_repository", forbidden)
    service = DataService(repository)
    first = service.snapshot(selected)
    second = service.snapshot(selected)
    assert first["id"] == second["id"]
    payload = first["payload"]
    assert len(payload["features"]) == 114
    assert not any(k.startswith(("target_", "next_")) for k in payload["features"])
    assert all(source_time(v["available_at"]) <= source_time(payload["as_of"]) for v in payload["features"].values())
    assert "labels" not in payload["input_source_hashes"]
    assert "tier1_core_primary_prospective" not in json.dumps(payload)
    assert "observed_sample_time_missing_reason" in payload["task"]


def test_future_anchor_and_admission_are_blocked(repository, selected, monkeypatch):
    frame = repository.frame.copy()
    column = next(
        s["tag"] + "__t_minus_0h"
        for s in repository.signals
        if np.isfinite(frame.loc[selected, s["tag"] + "__t_minus_0h"])
    )
    frame.loc[selected, column + "__age_h"] = -0.1
    monkeypatch.setattr(repository, "frame", frame)
    with pytest.raises(WorkbenchError, match="锚点"):
        DataService(repository).snapshot(selected)
    frame.loc[selected, column + "__age_h"] = 0.0
    admissions = {**repository.admissions, selected: dict(repository.admissions[selected])}
    admissions[selected]["feature_cutoff_at"] = (
        source_time(frame.loc[selected, "decision_at"]) + timedelta(seconds=1)
    ).isoformat()
    monkeypatch.setattr(repository, "admissions", admissions)
    with pytest.raises(WorkbenchError, match="准入卡"):
        DataService(repository).snapshot(selected)


def test_available_at_boundary_and_empty_recent_window():
    rows = pair()
    ledger = LabelLedger(rows)
    available = rows[0].available_at
    before = ledger.snapshot(available - timedelta(microseconds=1), minimum=1)["payload"]
    exact = ledger.snapshot(available, minimum=1)["payload"]
    assert before["labels"] == [] and before["status"] == "INSUFFICIENT_LABELS"
    assert exact["selected_events"] == 1 and len(exact["labels"]) == 2
    recent = ledger.snapshot(datetime(2026, 1, 1, tzinfo=UTC), minimum=1)["payload"]
    assert recent["labels"] == [] and recent["routing_action"] == "NONE"


def test_revision_replay_preserves_prior_evidence_and_rejects_conflicts():
    original = pair()
    ledger = LabelLedger(original)
    at = original[0].available_at
    previous = ledger.snapshot(at, minimum=1)
    revised = LabelRecord(
        **{
            **original[0].model_dump(),
            "revision": 2,
            "supersedes": original[0].record_id,
            "value": 3.0,
            "available_at": at + timedelta(days=1),
            "source_version": "synthetic-source-v2",
        }
    )
    updated = ledger.with_revision(revised)
    assert updated.snapshot(at, minimum=1)["id"] == previous["id"]
    current = updated.event_records(original[0].event_id, revised.available_at)
    assert current[0].value == 3.0 and current[0].revision == 2
    assert ledger.event_records(original[0].event_id, revised.available_at)[0].value == 2.0
    assert previous["payload"]["labels"][1]["value"] == 2.0
    with pytest.raises(WorkbenchError, match="修订"):
        ledger.with_revision(LabelRecord(**{**revised.model_dump(), "supersedes": "wrong-record"}))


def test_latest_invalid_revision_does_not_fall_back_to_old_good_label():
    original = pair()
    revised = LabelRecord(
        **{
            **original[0].model_dump(),
            "revision": 2,
            "supersedes": original[0].record_id,
            "available_at": original[0].available_at + timedelta(days=1),
            "value": None,
            "missing_reason": "withdrawn",
            "quality_eligible": False,
        }
    )
    result = LabelLedger(original).with_revision(revised).snapshot(revised.available_at, minimum=1)
    assert result["payload"]["labels"] == []


def test_comparisons_use_common_mature_events_and_report_missing_coverage():
    ledger = LabelLedger((*pair("synthetic-a", 1), *pair("synthetic-b", 2), *pair("synthetic-c", 3)))
    result = ledger.snapshot(
        datetime(2025, 1, 5, tzinfo=UTC),
        minimum=2,
        coverage={"A": {"synthetic-a", "synthetic-b"}, "B": {"synthetic-b", "synthetic-c"}},
    )["payload"]
    assert result["event_ids"] == ["synthetic-b"]
    assert result["coverage"] == {"A": {"available": 2, "missing": 1}, "B": {"available": 2, "missing": 1}}
    assert result["status"] == "INSUFFICIENT_LABELS" and result["routing_action"] == "NONE"


def test_private_dependencies_can_be_relocated_and_changes_are_detected(repository, tmp_path, monkeypatch):
    for folder, source in (
        ("inputs", repository.data_dir),
        ("contracts", repository.paths.contract_dir),
        ("labels", repository.label_dir),
    ):
        shutil.copytree(source, tmp_path / folder)
    monkeypatch.setenv("COPPER_MVP_DATA_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("COPPER_MVP_CONTRACT_DIR", str(tmp_path / "contracts"))
    monkeypatch.setenv("COPPER_MVP_LABEL_DIR", str(tmp_path / "labels"))
    relocated = DataRepository()
    assert relocated.dataset_version == repository.dataset_version
    np.testing.assert_allclose(relocated.X, repository.X, equal_nan=True)
    service = DataService(relocated)
    assert all("local_path" not in item for item in service.dependencies()["payload"]["resources"])
    assert service.labels.records
    index = tmp_path / "inputs/training_evaluation_index_v2.csv"
    with index.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(WorkbenchError, match="来源已改变"):
        service.snapshot(repository.frame.index[0])
    with pytest.raises(WorkbenchError, match="来源已改变"):
        service.labels.snapshot(datetime(2025, 12, 31, tzinfo=UTC))


def test_missing_labels_do_not_disable_prediction_snapshots(repository, selected, tmp_path):
    custom = DataRepository(label_dir=tmp_path / "absent")
    service = DataService(custom)
    assert service.snapshot(selected)["payload"]["features"]
    assert not service.dependencies()["payload"]["ready"]
    with pytest.raises(WorkbenchError, match="标签账本"):
        service.labels


def test_api_replay_label_gate_and_legacy_compatibility(repository, selected, tmp_path):
    with TestClient(create_app(tmp_path, repository, enforce_auth=False)) as client:
        assert client.get("/api/experiments").status_code == 200
        old = client.get("/api/events/" + selected).json()
        base = "/api/v2/data/events/" + selected
        task = client.get(base + "/task-spec")
        assert task.status_code == 200
        initial = client.get(base + "/evidence").json()["payload"]
        assert initial["labels"] == [] and initial["evaluation"]["status"] == "LABELS_NOT_AVAILABLE"
        assert initial["prediction"]["payload"]["predictions"]["cu"]["value"] == old["cu"]
        records = client.app.state.workbench._g1_data_service.labels.records
        available = max(r.available_at for r in records if r.event_id == selected)
        mature = client.get(base + "/evidence", params={"as_of": available.isoformat()}).json()["payload"]
        assert mature["evaluation"]["status"] == "EVALUABLE"
        assert len(mature["labels"]) == 2
        assert mature["prediction"]["payload"]["online_prediction_claim"] is False
        assert client.get(base + "/evidence", params={"as_of": "2024-01-01T00:00:00Z"}).status_code == 400
        assert client.get(base + "/evidence", params={"model_profile": "auto"}).status_code == 422
        assert client.get(base + "/snapshot", params={"target_cu_g_l": "100"}).status_code == 422
        assert (
            client.get("/api/v2/data/labels", params={"as_of": available.replace(tzinfo=None).isoformat()}).status_code
            == 422
        )
        assert client.get("/api/v2/data/dependencies", params={"local_path": str(tmp_path)}).status_code == 422
        assert client.get("/api/v2/data/dependencies", headers={"Origin": "https://example.com"}).status_code == 403
        assert client.get("/api/events/" + selected).json() == old


def test_feature_flag_disables_only_g1_routes(repository, tmp_path, monkeypatch):
    monkeypatch.setenv("COPPER_MVP_G1_ENABLED", "false")
    with TestClient(create_app(tmp_path, repository, enforce_auth=False)) as client:
        assert client.get("/api/v2/data/dependencies").status_code == 404
        assert client.get("/api/overview").status_code == 200


def test_ridge_preprocessing_matches_its_actual_training_fold(repository, selected):
    manager = ModelManager(DEFAULT_RUNS_DIR, repository)
    if not manager.catalog():
        pytest.skip("No saved local bundle; full training is exercised by test_workbench.")
    resolved = manager.resolve(selected, "DeltaRidge", "oof_replay")
    train = repository.X.loc[repository.training_ids(resolved["fold_id"])].to_numpy(float)
    numeric = train[:, :110]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        expected_impute = np.nanmedian(numeric, axis=0)
    expected_impute = np.nan_to_num(expected_impute, nan=0.0)
    filled = np.where(np.isnan(numeric), expected_impute, numeric)
    for model in resolved["models"].values():
        pipeline = model.named_steps["preprocess"].named_transformers_["numeric"]
        np.testing.assert_allclose(pipeline.named_steps["impute"].statistics_, expected_impute)
        np.testing.assert_allclose(pipeline.named_steps["scale"].mean_, filled.mean(axis=0), rtol=1e-9, atol=1e-8)
        before = pipeline.named_steps["scale"].mean_.copy()
        model.predict(np.nan_to_num(repository.X.loc[[selected]].to_numpy(float)) * 10)
        np.testing.assert_array_equal(pipeline.named_steps["scale"].mean_, before)


def test_training_uses_available_time_instead_of_recorded_time(repository, tmp_path):
    for name in ("outcome_ledger_v2.csv", "event_pair_index_v2.csv"):
        shutil.copyfile(repository.label_dir / name, tmp_path / name)
    fold = sorted(repository.cv.fold_id.unique())[0]
    train = repository.cv[repository.cv.fold_id.eq(fold) & repository.cv.fold_role.eq("TRAIN")]
    path = tmp_path / "event_pair_index_v2.csv"
    pairs = pd.read_csv(path)
    cutoff = pd.Timestamp(train.fold_fit_cutoff_at.iloc[0])
    pairs.loc[pairs.pair_id.eq(train.pair_id.iloc[0]), "target_available_at"] = (
        cutoff + pd.Timedelta(seconds=1)
    ).strftime("%Y-%m-%d %H:%M:%S")
    pairs.to_csv(path, index=False)
    copied = DataRepository(label_dir=tmp_path)
    with pytest.raises(WorkbenchError, match="可用时间"):
        copied.training_data()


def test_external_label_year_is_blocked_before_reading_outcome_values(repository, tmp_path, monkeypatch):
    pairs = pd.read_csv(repository.label_dir / "event_pair_index_v2.csv")
    pairs.loc[0, "target_available_at"] = "2026-01-01 00:00:00"
    pairs.to_csv(tmp_path / "event_pair_index_v2.csv", index=False)
    (tmp_path / "outcome_ledger_v2.csv").write_text("DO_NOT_READ", encoding="utf-8")
    copied = DataRepository(label_dir=tmp_path)
    original_read = pd.read_csv

    def guarded_read(path, *args, **kwargs):
        if str(path).endswith("outcome_ledger_v2.csv"):
            raise AssertionError("external outcomes were accessed")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read)
    with pytest.raises(WorkbenchError, match="年份边界"):
        copied.training_data()
    with pytest.raises(WorkbenchError, match="年份边界"):
        DataService(copied).labels


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_label_numeric_contract_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        LabelRecord(**{**pair()[0].model_dump(), "value": value})
