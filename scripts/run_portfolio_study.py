"""Run fixed and portfolio optimizers on predeclared mathematical or historical cases."""

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
    parser.add_argument("--mode", choices=("plant", "benchmark"), default="plant")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--request-key", required=True)
    parser.add_argument("--events", nargs="*")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20260911, 20260921)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cases <= 20:
        parser.error("cases must be between 1 and 20")
    import numpy as np

    from copper_mvp.access import Principal
    from copper_mvp.common import digest
    from copper_mvp.data import DataRepository
    from copper_mvp.modeling import ModelManager
    from copper_mvp.portfolio_comparison import PortfolioRequest
    from copper_mvp.portfolio_studies import PortfolioStudies

    data = DataRepository() if args.mode == "plant" else None
    models = ModelManager(args.run_dir, data) if data else None
    events = args.events or []
    if data and not events:
        pool = data.events(scope="dual", limit=5000)["items"]
        events = [
            pool[i]["event_id"]
            for i in np.unique(np.linspace(0, len(pool) - 1, min(args.cases, len(pool))).astype(int))
        ]
    request = PortfolioRequest(
        request_key=args.request_key,
        mode=args.mode,
        event_ids=tuple(events),
        seeds=tuple(args.seeds),
        total_budget=args.budget,
    )
    service = PortfolioStudies(args.run_dir, data, models)
    actor = Principal("owner", "owner")
    identifier = digest([actor.project_id, actor.user_id, request.request_key])[:32]
    print(
        json.dumps(
            {"id": identifier, "mode": args.mode, "cases": len(events) if data else 1, "seeds": len(args.seeds)}
        ),
        flush=True,
    )
    result = service.resume(actor, identifier) if args.resume else service.submit(actor, request)
    print(json.dumps({k: v for k, v in result.items() if k != "result"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)
