"""Run a complete nested ensemble comparison using the existing local data."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--request-key", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    args = parser.parse_args()
    from copper_mvp.access import Principal
    from copper_mvp.common import digest
    from copper_mvp.data import DataRepository
    from copper_mvp.ensemble_studies import EnsembleRequest, EnsembleStudies

    actor = Principal("owner", "owner")
    service = EnsembleStudies(args.run_dir, DataRepository())
    identifier = digest([actor.project_id, actor.user_id, args.request_key])[:32]
    print(json.dumps({"id": identifier, "status": "starting"}), flush=True)
    request = EnsembleRequest(
        comparison_id=args.comparison_id, request_key=args.request_key, max_wall_seconds=args.max_wall_seconds
    )
    result = service.resume(actor, identifier) if args.resume else service.submit(actor, request)
    print(json.dumps({k: v for k, v in result.items() if k != "result"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)
