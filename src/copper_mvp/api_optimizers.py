"""Read registered optimization methods and locally completed comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from copper_mvp.model_comparisons import process_alive
from copper_mvp.optimizer_registry import OPTIMIZERS, optimizer_catalog


def optimizer_router():
    router = APIRouter(prefix="/api/v2", tags=["Optimizer comparisons"])

    def root(request):
        return request.app.state.workbench.root / "optimizer_comparisons"

    def folder(request, run_id):
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise HTTPException(404, "没有该比较")
        path = root(request) / run_id
        if not (path / "state.json").is_file():
            raise HTTPException(404, "没有该比较")
        return path

    def state(path):
        record = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if record["status"] == "running" and not process_alive(record.get("owner_pid")):
            record["status"] = "interrupted"
        return record

    @router.get("/optimizers")
    def optimizers():
        return optimizer_catalog()

    @router.get("/optimizer-comparisons")
    def comparisons(request: Request):
        return {"items": sorted([state(p.parent) for p in root(request).glob("*/state.json")],
                                key=lambda item: item["created_at"], reverse=True)}

    @router.get("/optimizer-comparisons/{run_id}")
    def comparison(request: Request, run_id: str):
        path = folder(request, run_id)
        record = state(path)
        if (path / "comparison.json").is_file():
            result = json.loads((path / "comparison.json").read_text(encoding="utf-8"))
            result["results"] = [{k: v for k, v in row.items() if k != "candidates"} for row in result["results"]]
            record["result"] = result
        return record

    @router.get("/optimizer-comparisons/{run_id}/cases/{case}/{optimizer}/{seed}")
    def run_result(request: Request, run_id: str, case: int, optimizer: str, seed: int):
        if not 0 <= case < 20 or optimizer not in OPTIMIZERS or not 0 <= seed < 2**31:
            raise HTTPException(404, "没有该运行")
        path = folder(request, run_id) / f"case_{case}" / str(seed) / optimizer / "result.json"
        if not path.is_file():
            raise HTTPException(404, "没有该运行")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.get("/optimizer-comparisons/{run_id}/export")
    def export(request: Request, run_id: str, format: str = "md"):
        if format not in ("md", "csv", "json"):
            raise HTTPException(422, "只支持 md、csv、json")
        path = folder(request, run_id) / {"md": "report.md", "csv": "metrics.csv", "json": "comparison.json"}[format]
        if not path.is_file():
            raise HTTPException(409, "比较尚未完成")
        return FileResponse(path, filename=f"optimizer-comparison-{run_id}.{format}")

    return router
