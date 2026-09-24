"""Run a preregistered temporal calibration comparison on local development data."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    from copper_mvp.access import Principal
    from copper_mvp.calibration_contracts import CalibrationRequest
    from copper_mvp.calibration_service import CalibrationService
    from copper_mvp.data import DataRepository

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-key", required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/mvp")
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    args = parser.parse_args()
    if not args.run_root.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise RuntimeError("Calibration output must remain inside the project runs directory")
    values = {"request_key": args.request_key, "max_wall_seconds": args.max_wall_seconds}
    if args.methods:
        values["methods"] = args.methods
    if args.seeds:
        values["seeds"] = args.seeds
    request = CalibrationRequest(**values)
    service = CalibrationService(args.run_root, DataRepository())
    actor = Principal("owner", "owner")
    with ThreadPoolExecutor(max_workers=1) as executor:
        state = service.submit(actor, request, executor)
        print(json.dumps({"study_id": state["id"], "status": state["status"], "reused": state["reused"]}), flush=True)
    result = service.get(actor, state["id"])
    print(
        json.dumps(
            {
                "study_id": state["id"],
                "status": result["status"],
                "common_events": result.get("result", {}).get("common_events"),
                "error": result.get("error"),
            }
        ),
        flush=True,
    )
    if result["status"] != "completed":
        sys.exit(1)
