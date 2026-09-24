"""Run a local fixed-protocol shadow routing study from an existing model comparison."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--request-key", required=True)
    args = parser.parse_args()
    from copper_mvp.access import Principal
    from copper_mvp.common import digest
    from copper_mvp.data import DataRepository
    from copper_mvp.routing_studies import RoutingRequest, RoutingStudies

    service = RoutingStudies(args.run_dir, DataRepository())
    print(
        json.dumps({"id": digest(["copper-research", "owner", args.request_key])[:32], "status": "starting"}),
        flush=True,
    )
    result = service.submit(
        Principal("owner", "owner"), RoutingRequest(comparison_id=args.comparison_id, request_key=args.request_key)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "result"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)
