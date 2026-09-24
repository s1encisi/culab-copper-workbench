"""Necessary ensemble checks: convexity, temporal OOF, persistence and future labels."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from copper_mvp.data_contracts import LabelRecord
from copper_mvp.ensemble_models import EnsembleSettings, simplex_weights
from copper_mvp.ensemble_time import EnsembleData, block_sample
from copper_mvp.ensemble_training import TrainingBudget, train_outer_fold
from copper_mvp.labels import LabelLedger


def fixture_data():
    rng = np.random.default_rng(91)
    events = [f"ensemble-{i:03d}" for i in range(160)]
    begin = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [begin + timedelta(hours=8 * i) for i in range(len(events))]
    X = rng.normal(size=(len(events), 114))
    X[:, 0] = 40 + np.arange(len(events)) * 0.02 + rng.normal(scale=0.05, size=len(events))
    X[:, 1] = 1000 + np.arange(len(events)) * 2 + rng.normal(scale=3, size=len(events))
    X[:, 110:] = rng.integers(0, 3, size=(len(events), 4))
    frame = pd.DataFrame({"decision_at": dates}, index=events)
    values = pd.DataFrame(X, index=events)
    data = SimpleNamespace(
        X=values,
        row=lambda e: frame.loc[e],
        mode_cards={e: {"mode_code": "A" if i % 3 else "B"} for i, e in enumerate(events)},
    )
    labels = []
    for i, e in enumerate(events):
        actual = X[min(i + 1, len(events) - 1), :2]
        for t, (target, unit) in enumerate((("cu", "g/L"), ("as", "mg/L"))):
            labels.append(
                LabelRecord(
                    event_id=e,
                    pair_id="pair-" + e,
                    target=target,
                    value=actual[t],
                    unit=unit,
                    decision_at=dates[i],
                    available_at=dates[i] + timedelta(hours=8),
                    assumed_sample_time=dates[i] + timedelta(hours=6),
                    quality_eligible=True,
                    source_version="synthetic-ensemble",
                    source_kind="synthetic_fixture",
                )
            )
    fold = {
        "fold_id": "FOLD_SYNTHETIC",
        "fit_cutoff_at": dates[120].isoformat(),
        "train_event_ids": events[:120],
        "validation_event_ids": events[120:144],
    }
    settings = EnsembleSettings(
        inner_boundaries=(0.5, 0.75, 1.0),
        ridge_alphas=(1.0,),
        convex_penalties=(0.0, 0.1),
        dynamic_windows=(8, 16),
        dynamic_minimum=5,
        bag_members=(2, 3),
    )
    return data, LabelLedger(labels), fold, settings


def test_simplex_solution_is_nonnegative_normalized_and_beats_vertices():
    P = np.array([[0.0, 2.0, 4.0], [0.0, 4.0, 2.0], [0.0, 3.0, 3.0], [0.0, 2.0, 2.0]])
    actual = P @ np.array([0.25, 0.5, 0.25])
    weights = simplex_weights(P, actual, 1.0)
    assert np.min(weights) >= 0 and weights.sum() == pytest.approx(1)
    assert np.mean((P @ weights - actual) ** 2) < 1e-12
    regularized = simplex_weights(P, actual, 1.0, 100.0)
    assert regularized[0] > weights[0]


def test_nested_plan_uses_only_available_label_versions_and_reproducible_blocks():
    data, ledger, fold, settings = fixture_data()
    original = next(r for r in ledger.records if r.event_id == "ensemble-005" and r.target == "cu")
    late = original.model_copy(
        update={
            "revision": 2,
            "supersedes": original.record_id,
            "available_at": data.row("ensemble-080").decision_at,
            "value": original.value + 1,
        }
    )
    book = EnsembleData(data, ledger.with_revision(late))
    plan = book.inner_plan(fold["train_event_ids"], fold["fit_cutoff_at"], settings)
    assert plan[0]["fit_cutoff_at"] < late.available_at.isoformat()
    for part in plan:
        assert set(part["train_event_ids"]).isdisjoint(part["validation_event_ids"])
        assert part["latest_training_decision"] < part["fit_cutoff_at"]
        assert part["latest_label_available_at"] <= part["fit_cutoff_at"]
    assert original.record_id in book.label_ids(plan[0]["train_event_ids"], plan[0]["fit_cutoff_at"])
    assert late.record_id not in book.label_ids(plan[0]["train_event_ids"], plan[0]["fit_cutoff_at"])
    ids = plan[0]["train_event_ids"]
    dates = [book.decision(e) for e in ids]
    first, proof = block_sample(ids, dates, 7, 19)
    second, other = block_sample(ids, dates, 7, 19)
    assert np.array_equal(first, second) and proof == other
    assert first.max() < len(ids)


def test_real_nested_fit_serializes_and_future_labels_do_not_change_earlier_predictions(tmp_path):
    data, ledger, fold, settings = fixture_data()
    book = EnsembleData(data, ledger)
    first = tmp_path / "first"
    result = train_outer_fold(book, fold, settings, (19,), 20260905, first, TrainingBudget(180, lambda value: None))
    from copper_mvp.ensemble_evaluation import verify_fold

    assert verify_fold(first, result, book, fold)
    assert result["inner_oof_events"] == 60 and result["warmup_events"] == 60
    table = pd.read_csv(first / "predictions.csv")
    expected = {
        "Persistence",
        "DeltaRidge",
        "DeltaHGB",
        "Stacking",
        "OOFConvex",
        "Blending",
        "MeanEnsemble",
        "DynamicConvex",
        "Bagging_d1",
        "Bagging_d7",
        "BlockBagging",
    }
    assert set(table.method_id) == expected and len(table) == 24 * len(expected)
    for entry in result["artifacts"]:
        if entry["artifact_type"] != "static_predictor":
            continue
        model = joblib.load(first / entry["path"])
        predicted = model.predict(book.X(fold["validation_event_ids"]))
        method = entry.get("method", entry["name"])
        rows = table[table.method_id.eq(method)][["cu", "as"]].to_numpy()
        assert np.allclose(predicted, rows, atol=1e-9, rtol=1e-10)
    changed = LabelLedger(
        [r.model_copy(update={"value": r.value + 1000}) if r.event_id >= "ensemble-134" else r for r in ledger.records]
    )
    second = tmp_path / "second"
    train_outer_fold(
        EnsembleData(data, changed), fold, settings, (19,), 20260905, second, TrainingBudget(180, lambda value: None)
    )
    after = pd.read_csv(second / "predictions.csv")
    early = table.event_id < "ensemble-134"
    assert np.allclose(table.loc[early, ["cu", "as"]], after.loc[early, ["cu", "as"]], atol=1e-9, rtol=1e-10)
    static = table.method_id != "DynamicConvex"
    assert np.allclose(table.loc[static, ["cu", "as"]], after.loc[static, ["cu", "as"]], atol=1e-9, rtol=1e-10)
    first_selection = json.loads((first / "selection.json").read_text(encoding="utf-8"))
    second_selection = json.loads((second / "selection.json").read_text(encoding="utf-8"))
    assert first_selection == second_selection
    with (first / "weight_decisions.jsonl").open() as stream:
        for line in stream:
            decision = json.loads(line)
            assert (
                decision["latest_label_available_at"] is None
                or decision["latest_label_available_at"] <= decision["as_of"]
            )
            assert np.sum(decision["weights"]) == pytest.approx(1)


def test_independent_evaluation_keeps_all_seeds_and_checks_hashes(tmp_path, monkeypatch):
    from copper_mvp.common import WorkbenchError, file_hash, write_json
    from copper_mvp.ensemble_evaluation import evaluate_ensembles

    data, ledger, fold, settings = fixture_data()
    data.fold_for = dict.fromkeys(fold["validation_event_ids"], fold["fold_id"])
    events = fold["validation_event_ids"]
    records = []
    for method, seeds, error in (
        ("Persistence", [-1], 2.0),
        ("DeltaRidge", [-1], 1.0),
        ("DeltaHGB", [-1], 0.5),
        ("BlockBagging", [11, 12], 0.25),
    ):
        for seed in seeds:
            for i, event in enumerate(events):
                actual = [
                    next(r.value for r in ledger.records if r.event_id == event and r.target == t) for t in ("cu", "as")
                ]
                offset = error if seed != 12 else error + 0.1
                records.append(
                    {
                        "event_id": event,
                        "fold_id": fold["fold_id"],
                        "method_id": method,
                        "seed": None if seed == -1 else seed,
                        "decision_at": data.row(event).decision_at.isoformat(),
                        "fit_cutoff_at": fold["fit_cutoff_at"],
                        "cu": actual[0] + offset,
                        "as": actual[1] + offset,
                        "status": "completed",
                    }
                )
    root = tmp_path
    protocol = {
        "source": {"fixture": "synthetic"},
        "base_methods": ["Persistence", "DeltaRidge", "DeltaHGB"],
        "expected_methods": {"Persistence": [-1], "DeltaRidge": [-1], "DeltaHGB": [-1], "BlockBagging": [11, 12]},
        "request": {"seeds": [11, 12]},
        "evaluation_as_of": "2025-01-01T00:00:00+00:00",
    }
    write_json(root / "protocol.json", protocol)
    pd.DataFrame(records).to_csv(root / "predictions.csv", index=False)
    (root / "decisions.jsonl").write_bytes(b"")
    manifest = {
        "protocol_sha256": file_hash(root / "protocol.json"),
        "predictions_sha256": file_hash(root / "predictions.csv"),
        "decisions_sha256": file_hash(root / "decisions.jsonl"),
        "folds": [],
        "training_resources": {},
    }
    write_json(root / "replay_manifest.json", manifest)
    monkeypatch.setattr("copper_mvp.ensemble_evaluation.DataService", lambda value: SimpleNamespace(labels=ledger))
    monkeypatch.setattr("copper_mvp.ensemble_evaluation.source_signature", lambda value: {"fixture": "synthetic"})
    result = evaluate_ensembles(data, root)
    metric = next(r for r in result["metrics"] if r["method_id"] == "BlockBagging" and r["target"] == "cu")
    assert metric["mae"] == pytest.approx(0.3) and metric["seed_count"] == 2
    assert metric["seed_mae_sd"] > 0 and result["common_events"] == 24
    assert metric["paired_ci95"]["status"] == "INSUFFICIENT_BLOCKS"
    (root / "predictions.csv").write_bytes(b"tampered")
    with pytest.raises(WorkbenchError, match="哈希"):
        evaluate_ensembles(data, root)


def test_resume_is_once_and_never_reopens_a_running_study(tmp_path):
    from pathlib import Path

    from copper_mvp.access import Principal
    from copper_mvp.common import WorkbenchError, digest, file_hash, write_json
    from copper_mvp.ensemble_studies import EnsembleRequest, EnsembleStudies

    service = EnsembleStudies(tmp_path, None)
    request = EnsembleRequest(request_key="resume-fixture", comparison_id="f" * 32, seeds=(11,))
    actor = Principal("owner", "owner")
    identifier = digest([actor.project_id, actor.user_id, request.request_key])[:32]
    folder = service.directory(identifier)
    folder.mkdir()
    service.comparisons = SimpleNamespace(get=lambda *args, **kwargs: {"fingerprint": "synthetic-source"})
    package = Path(__file__).resolve().parents[2] / "src" / "copper_mvp"
    fingerprint = digest(
        {
            "request": request.model_dump(),
            "source": "synthetic-source",
            "policy": service.policy_spec(),
            "code": {n: file_hash(package / n) for n in service.signature_files},
        }
    )
    write_json(
        folder / "state.json",
        {
            "id": identifier,
            "owner_id": "owner",
            "project_id": actor.project_id,
            "status": "failed",
            "request": request.model_dump(mode="json"),
            "fingerprint": fingerprint,
            "created_at": "synthetic",
        },
    )
    queue = []
    executor = SimpleNamespace(submit=lambda *args: queue.append(args))
    service.resume(actor, identifier, executor)
    assert len(queue) == 1
    with pytest.raises(WorkbenchError, match="只恢复"):
        service.resume(actor, identifier, executor)
    assert len(queue) == 1
