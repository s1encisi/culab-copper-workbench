"""Run the registered optimizers with one shared, fully counted problem."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from copper_mvp.common import DEFAULT_RUNS_DIR, PROJECT_ROOT, digest, utc_now, write_json
from copper_mvp.data import DataRepository
from copper_mvp.modeling import ModelManager
from copper_mvp.optimizer_comparison import compare_optimizers
from copper_mvp.optimizer_registry import LEGACY_OPTIMIZERS, OPTIMIZERS, OptimizerComparisonRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plant", "benchmark"), default="plant")
    parser.add_argument("--events", nargs="*")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260911, 20260912, 20260913])
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(LEGACY_OPTIMIZERS))
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--request-key", default="g2b-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUNS_DIR / "optimizer_comparisons")
    args = parser.parse_args()
    if not 1 <= args.cases <= 20:
        parser.error("cases must be between 1 and 20")
    root = args.output_root.resolve()
    if not root.is_relative_to((PROJECT_ROOT / "runs").resolve()):
        parser.error("Private optimization results must remain under local runs/")
    data = DataRepository() if args.mode == "plant" else None
    models = ModelManager(DEFAULT_RUNS_DIR, data) if data else None
    events = args.events or []
    if data and not events:
        pool = data.events(scope="dual", limit=5000)["items"]
        events = [pool[i]["event_id"] for i in np.unique(np.linspace(0, len(pool)-1, min(args.cases, len(pool))).astype(int))]
    request = OptimizerComparisonRequest(request_key=args.request_key, mode=args.mode, event_ids=tuple(events),
              optimizers=tuple(args.optimizers), seeds=tuple(args.seeds), total_budget=args.budget, seconds_per_run=args.seconds)
    run_id = digest(request.request_key)[:32]
    output = root / run_id
    if output.exists():
        parser.error("This request directory already exists; use a new request key.")
    output.mkdir(parents=True)
    state = {"run_id": run_id, "status": "running", "created_at": utc_now(), "owner_pid": os.getpid(),
             "request": request.model_dump(mode="json")}
    write_json(output / "state.json", state)
    print(json.dumps({"run_id": run_id, "output": str(output)}, ensure_ascii=False), flush=True)
    def progress(detail):
        state["progress"] = detail
        write_json(output / "state.json", state)
        if detail.get("phase") == "running":
            print(json.dumps(detail, ensure_ascii=False), flush=True)
    try:
        result = compare_optimizers(data, models, request, output, progress)
        state.update(status="completed", finished_at=utc_now(), runs=result["runs"])
        write_json(output / "state.json", state)
    except Exception as exc:
        state.update(status="failed", finished_at=utc_now(), error_code=getattr(exc, "code", type(exc).__name__))
        write_json(output / "state.json", state)
        raise
    print(json.dumps({"status": result["status"], "runs": result["runs"], "budget_max": max(r["total_evaluations"] for r in result["results"])}), flush=True)
    print("Report: " + str(output / "report.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
