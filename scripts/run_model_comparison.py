"""Run a local fixed-protocol comparison and print its progress/report path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copper_mvp.common import DEFAULT_RUNS_DIR, PROJECT_ROOT
from copper_mvp.data import DataRepository
from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_registry import LEGACY_METHOD_IDS, METHOD_IDS, SEED, ComparisonRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request-key", default="g2a-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    parser.add_argument("--methods", nargs="+", choices=METHOD_IDS, default=list(LEGACY_METHOD_IDS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-wall-seconds", type=int, default=900)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUNS_DIR / "model_comparisons")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to((PROJECT_ROOT / "runs").resolve()):
        parser.error("Private comparison artifacts must remain under the local runs/ directory.")
    request = ComparisonRequest(
        request_key=args.request_key,
        methods=tuple(args.methods),
        seed=args.seed,
        max_wall_seconds=args.max_wall_seconds,
    )
    service = ComparisonService(root, DataRepository())
    with ThreadPoolExecutor(max_workers=1) as executor:
        state = service.submit(request, executor)
        run_id = state["run_id"]
        print(
            json.dumps({"run_id": run_id, "reused": state["reused"], "root": str(service.directory(run_id))}),
            flush=True,
        )
        previous = None
        while True:
            state = service.get(run_id, result=False)
            progress = state.get("progress")
            if progress != previous:
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                previous = progress
            if state["status"] not in ("queued", "running"):
                break
            time.sleep(0.25)
    state = service.get(run_id)
    if state["status"] != "completed":
        print(json.dumps({"status": state["status"], "error": state.get("error")}, ensure_ascii=False))
        return 1
    result = state["result"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "common_events": result["common_events"],
                "best_observed_mae": result["best_observed_mae"],
                "automatic_promotion": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("Report: " + str(service.directory(run_id) / "report.md"), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
