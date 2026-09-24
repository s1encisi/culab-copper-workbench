import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from copper_mas.contracts.cards import ProcessModeCardV2
from copper_mas.models.predictors import (
    CandidateBundlePredictor,
    PersistencePredictor,
    load_selected_predictor,
)


def _mode() -> ProcessModeCardV2:
    return ProcessModeCardV2(
        origin_event_id="evt-1",
        decision_at=datetime(2025, 1, 1),
        mode_code="TEST",
        confidence=1,
        evidence_observation_ids=(),
    )


def test_persistence_predictor_uses_current_results_only() -> None:
    result = PersistencePredictor()({"origin_cu_g_l": 4.2, "origin_as_mg_l": 1500}, _mode())
    assert result.predicted_cu_g_l == 4.2
    assert result.predicted_as_mg_l == 1500


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(path: Path) -> None:
    path.mkdir()
    x = pd.DataFrame({"origin_cu_g_l": [1.0, 2.0], "origin_as_mg_l": [10.0, 20.0]})
    artifacts = []
    for target, constant in (("target_cu_g_l", 3.0), ("target_as_mg_l", 30.0)):
        model = DummyRegressor(strategy="constant", constant=constant).fit(x, [constant, constant])
        filename = f"dummy__{target}.joblib"
        joblib.dump(model, path / filename)
        artifacts.append(
            {
                "artifact": filename,
                "model_name": "Dummy",
                "target_name": target,
                "sha256": _hash(path / filename),
            }
        )
    names = ["origin_cu_g_l", "origin_as_mg_l"]
    manifest = {
        "bundle_id": "test-bundle",
        "bundle_status": "FROZEN_DEVELOPMENT_CANDIDATES_NO_WINNER",
        "external_2026_used": False,
        "features": {
            "known_at_decision_only": True,
            "ordered_names": names,
            "ordered_names_sha256": hashlib.sha256(
                json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "serialized_candidates": artifacts,
    }
    (path / "bundle_manifest_v1.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_candidate_loader_verifies_and_predicts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    predictor = CandidateBundlePredictor(bundle, cu_model_name="Dummy", as_model_name="Dummy")
    result = predictor({"origin_cu_g_l": 4.0, "origin_as_mg_l": 40.0}, _mode())
    assert result.predicted_cu_g_l == 3.0
    assert result.predicted_as_mg_l == 30.0


def test_candidate_loader_refuses_tampered_model(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    with (bundle / "dummy__target_cu_g_l.joblib").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        CandidateBundlePredictor(bundle, cu_model_name="Dummy", as_model_name="Dummy")


def test_selected_predictor_requires_explicit_development_only_manifest(tmp_path: Path) -> None:
    manifest = {
        "external_2026_read": False,
        "selected_per_target": {
            "target_cu_g_l": {"model_name": "Persistence"},
            "target_as_mg_l": {"model_name": "Persistence"},
        },
        "a4_predictor": {"model_id": "PERSISTENCE_CURRENT_RESULT_V1"},
    }
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert isinstance(load_selected_predictor(path), PersistencePredictor)
    manifest["external_2026_read"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="external_2026_read=false"):
        load_selected_predictor(path)
