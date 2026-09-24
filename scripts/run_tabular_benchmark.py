"""Run the fixed tabular neural study, then its matched MLP controls."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"] = "false"
if __name__ == "__main__":
    from copper_mvp.common import PROJECT_ROOT
    from copper_mvp.data import DataRepository
    from copper_mvp.model_inventory_study import run_inventory_study
    from copper_mvp.tabular_controls import run_matched_controls
    from copper_mvp.tabular_registry import TABULAR_METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-key", required=True)
    parser.add_argument("--reference-comparison-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/mvp"))
    args = parser.parse_args()
    if not args.run_root.resolve().is_relative_to((PROJECT_ROOT / "runs").resolve()):
        parser.error("Private outputs must remain under local runs/")
    data = DataRepository()
    seeds = list(range(20260911, 20260916))
    emit = lambda value: print(json.dumps(value, ensure_ascii=False), flush=True)
    root, result = run_inventory_study(
        data,
        args.run_root,
        args.reference_comparison_id,
        args.request_key,
        TABULAR_METHODS,
        seeds,
        progress=emit,
        max_wall_seconds=21600,
    )
    if result["status"] != "completed":
        raise RuntimeError("Neural study incomplete; controls not started")
    emit({"neural_study": root.name, "status": result["status"]})
    control_root, control = run_matched_controls(
        data, args.run_root, root.name, args.request_key + "-mlp-controls", progress=emit
    )
    emit({"status": control["status"], "neural_study": root.name, "control_study": control_root.name})
