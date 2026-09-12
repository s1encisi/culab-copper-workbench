"""Run a multi-seed inventory comparison on authorized local development data."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

if __name__ == "__main__":
    from copper_mvp.common import PROJECT_ROOT, digest
    from copper_mvp.data import DataRepository
    from copper_mvp.model_inventory_study import run_inventory_study
    from copper_mvp.model_registry import METHOD_IDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-key", required=True)
    parser.add_argument("--reference-comparison-id", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHOD_IDS,
                        default=["CatBoost", "NGBoost", "EBM", "Cubist"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260911, 20260916)))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mvp"))
    args = parser.parse_args()
    if not 1 <= len(args.seeds) <= 5 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use one to five distinct seeds")
    if len(set(args.methods)) != len(args.methods) or "Persistence" in args.methods:
        parser.error("select distinct fitted methods; Persistence is included automatically")
    root = args.run_dir.resolve()
    if not root.is_relative_to((PROJECT_ROOT / "runs").resolve()):
        parser.error("results must remain under local runs/")
    print(json.dumps({"study_id": digest(args.request_key)[:32], "seeds": args.seeds}), flush=True)
    folder, result = run_inventory_study(DataRepository(), root, args.reference_comparison_id,
        args.request_key, args.methods, args.seeds,
        progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    print(json.dumps({"status": result["status"], "common_events": result["common_events"],
                      "output": str(folder)}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 1)
