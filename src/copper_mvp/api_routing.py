"""Authenticated routing study creation and evidence lookup."""

import threading

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from copper_mvp.common import WorkbenchError
from copper_mvp.routing_studies import RoutingRequest, RoutingStudies


def routing_router():
    router = APIRouter(prefix="/api/v2/prediction-studies")
    lock = threading.Lock()

    def service(request):
        wb = request.app.state.workbench
        with lock:
            if not hasattr(wb, "_routing_studies"):
                wb._routing_studies = RoutingStudies(wb.root, wb.data)
        return wb._routing_studies

    def actor(request, permission="read"):
        value = request.state.principal
        value.require(permission)
        return value

    @router.post("", status_code=202)
    def create(request: Request, payload: RoutingRequest):
        return service(request).submit(actor(request, "compute"), payload, request.app.state.workbench.executor)

    @router.get("")
    def history(request: Request):
        return {"items": service(request).list(actor(request))}

    @router.get("/{identifier}")
    def study(request: Request, identifier: str):
        return service(request).get(actor(request), identifier)

    @router.get("/{identifier}/decisions")
    def decisions(request: Request, identifier: str, event_id: str):
        return service(request).decisions(actor(request), identifier, event_id)

    @router.get("/{identifier}/export")
    def export(request: Request, identifier: str):
        current = service(request)
        state = current.get(actor(request), identifier)
        if state["status"] != "completed":
            raise WorkbenchError("回放尚未完成", "ROUTING_NOT_READY")
        return FileResponse(current.directory(identifier) / "report.md", filename="routing-study-" + identifier + ".md")

    return router
