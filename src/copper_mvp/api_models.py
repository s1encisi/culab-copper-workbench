"""G2a registered models and comparison APIs, separate from legacy model selection."""
from __future__ import annotations

import threading
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_registry import ComparisonPredictionRequest, ComparisonRequest, catalog


def model_router():
    router = APIRouter(prefix="/api/v2", tags=["G2a model comparisons"])
    lock = threading.Lock()

    def service(request):
        wb = request.app.state.workbench
        with lock:
            if not hasattr(wb, "_g2_comparisons"):
                wb._g2_comparisons = ComparisonService(wb.root / "model_comparisons", wb.data)
        return wb._g2_comparisons

    def require_no_query(request):
        if request.query_params:
            raise HTTPException(422, detail="此接口不接受额外查询参数")

    @router.get("/models")
    def models(request: Request):
        require_no_query(request)
        return catalog()

    @router.post("/model-comparisons", status_code=202)
    def create(request: Request, payload: ComparisonRequest):
        require_no_query(request)
        return service(request).submit(payload, request.app.state.workbench.executor)

    @router.get("/model-comparisons")
    def history(request: Request):
        require_no_query(request)
        return {"items": service(request).list()}

    @router.get("/model-comparisons/{run_id}")
    def comparison(request: Request, run_id: str):
        require_no_query(request)
        return service(request).get(run_id)

    @router.post("/model-comparisons/{run_id}/predict")
    def predict(request: Request, run_id: str, payload: ComparisonPredictionRequest):
        require_no_query(request)
        return service(request).predict(run_id, payload)

    @router.get("/model-comparisons/{run_id}/export")
    def export(request: Request, run_id: str, format: str = "md"):
        if set(request.query_params) - {"format"} or format not in ("md", "csv", "json"):
            raise HTTPException(422, detail="只支持 md、csv、json 导出")
        current = service(request)
        state = current.get(run_id)
        if state["status"] != "completed":
            raise HTTPException(409, detail="比较尚未完成")
        name = {"md": "report.md", "csv": "metrics.csv", "json": "evaluation.json"}[format]
        return FileResponse(current.directory(run_id) / name, filename=f"comparison-{run_id}.{format}")

    return router
