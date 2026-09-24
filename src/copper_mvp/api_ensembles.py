"""Authenticated nested ensemble training and artifact inspection."""

import threading

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from copper_mvp.common import WorkbenchError
from copper_mvp.ensemble_studies import EnsembleRequest, EnsembleStudies


def ensemble_router():
    router = APIRouter(prefix="/api/v2/ensemble-studies")
    lock = threading.Lock()

    def service(request):
        wb = request.app.state.workbench
        with lock:
            if not hasattr(wb, "_ensemble_studies"):
                wb._ensemble_studies = EnsembleStudies(wb.root, wb.data)
        return wb._ensemble_studies

    def actor(request, permission="read"):
        value = request.state.principal
        value.require(permission)
        return value

    @router.post("", status_code=202)
    def create(request: Request, payload: EnsembleRequest):
        return service(request).submit(actor(request, "compute"), payload, request.app.state.workbench.executor)

    @router.get("")
    def history(request: Request):
        return {"items": service(request).list(actor(request))}

    @router.get("/{identifier}")
    def study(request: Request, identifier: str):
        return service(request).get(actor(request), identifier)

    @router.post("/{identifier}/resume", status_code=202)
    def resume(request: Request, identifier: str):
        return service(request).resume(actor(request, "compute"), identifier, request.app.state.workbench.executor)

    @router.get("/{identifier}/decisions")
    def decisions(request: Request, identifier: str, event_id: str):
        return service(request).decisions(actor(request), identifier, event_id)

    @router.get("/{identifier}/export")
    def export(request: Request, identifier: str):
        current = service(request)
        if current.get(actor(request), identifier)["status"] != "completed":
            raise WorkbenchError("集成研究尚未完成", "ENSEMBLE_NOT_READY")
        return FileResponse(current.directory(identifier) / "report.md", filename="ensemble-" + identifier + ".md")

    return router
