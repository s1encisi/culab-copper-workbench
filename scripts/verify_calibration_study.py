"""Replay persisted C90 models and authenticated calibration prediction routes."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from copper_mvp.access import Principal
from copper_mvp.api import create_app
from copper_mvp.calibration import SplitC90
from copper_mvp.calibration_evaluation import read_calibration_training
from copper_mvp.calibration_service import CalibrationService, load_calibration_model
from copper_mvp.common import file_hash, write_json
from copper_mvp.data import DataRepository
from copper_mvp.neural_replay import NEURAL_METHODS, neural_replay_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/mvp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Choose a fresh verification output")
    args.output.mkdir(parents=True)
    data = DataRepository()
    service = CalibrationService(args.run_root, data)
    actor = Principal("owner", "owner")
    state = service.get(actor, args.study_id)
    assert state["status"] == "completed"
    folder = service.directory(args.study_id)
    manifest, protocol = read_calibration_training(folder)
    rows = pd.read_csv(folder / "predictions.csv")
    report = {"status": "running", "study_id": args.study_id, "artifacts": [], "replayed_events": 0, "api_cases": 0}
    try:
        for entry in manifest["artifacts"]:
            if entry["status"] != "completed" or entry["fold_id"] == "DEVELOPMENT":
                continue
            model_path = folder / entry["model_path"]
            assert file_hash(model_path) == entry["model_sha256"]
            model = load_calibration_model(model_path, entry["method_id"])
            part = rows[
                rows.fold_id.eq(entry["fold_id"]) & rows.method_id.eq(entry["method_id"]) & rows.seed.eq(entry["seed"])
            ]
            events = part.event_id.tolist()
            expected = part[["cu", "as"]].to_numpy(float)
            actual = model.predict_context(data, events)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)
            sample = np.unique(np.linspace(0, len(events) - 1, 3).astype(int))
            latencies = []
            for index in sample:
                start = perf_counter()
                result = service.predict(actor, args.study_id, events[index], entry["method_id"], entry["seed"])
                latencies.append((perf_counter() - start) * 1000)
                predicted = np.array([[result["point"]["cu"], result["point"]["as"]]])
                if entry["method_id"] in NEURAL_METHODS:
                    passed, _ = neural_replay_error(
                        predicted,
                        expected[[index]],
                        data.X.loc[[events[index]]].to_numpy(float)[:, :2],
                        model.fit_metadata["target_delta_scale"],
                    )
                    assert passed
                else:
                    np.testing.assert_allclose(predicted, expected[[index]], rtol=1e-10, atol=1e-9)
                calibrator = SplitC90.from_manifest(result["calibrator"])
                calculated = calibrator.intervals(predicted)
                for mode, bounds in calculated.items():
                    for j, target in enumerate(("cu", "as")):
                        assert result["intervals"][mode][target]["lower"] == float(bounds[0, 0, j])
                        assert result["intervals"][mode][target]["upper"] == float(bounds[0, 1, j])
                report["replayed_events"] += 1
            report["artifacts"].append(
                {
                    "method_id": entry["method_id"],
                    "seed": entry["seed"],
                    "fold_id": entry["fold_id"],
                    "full_batch_events": len(events),
                    "p50_api_ms": float(np.percentile(latencies, 50)),
                    "p95_api_ms": float(np.percentile(latencies, 95)),
                }
            )
            write_json(args.output / "verification.json", report)
        with TestClient(create_app(args.output / "api-runtime", data), base_url="http://127.0.0.1") as client:
            key = client.app.state.workbench.access.owner_key_path.read_text(encoding="utf-8").strip()
            assert client.post("/api/auth/session", json={"access_code": key}).status_code == 200
            client.app.state.workbench.calibrations = service
            response = client.get("/api/v2/calibrations/" + args.study_id)
            assert response.status_code == 200
            event = next(e for e in rows.event_id if e in data.fold_for)
            response = client.post(
                "/api/v2/calibrations/" + args.study_id + "/predict",
                json={"event_id": event, "method_id": "Persistence"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["evaluation_mode"] == "historical_oof_replay"
            viewer = client.post("/api/v2/access-keys", json={"user_id": "viewer", "role": "viewer"}).json()[
                "access_code"
            ]
            forbidden = client.post(
                "/api/v2/calibrations",
                json={"request_key": "unauthorized-fit"},
                headers={"Authorization": "Bearer " + viewer},
            )
            assert forbidden.status_code == 403
            report["api_cases"] = 3
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "verification.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "artifacts": len(report["artifacts"]),
                "replayed_events": report["replayed_events"],
                "api_cases": report["api_cases"],
            }
        )
    )


if __name__ == "__main__":
    main()
