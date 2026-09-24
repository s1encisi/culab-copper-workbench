"""Model lifecycle and explicit research-release endpoints."""

import threading

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from copper_mvp.model_lifecycle import ModelLifecycle
from copper_mvp.release_contracts import (
    ArtifactRequest,
    ForecastRequest,
    ReleaseApproval,
    ReleaseRequest,
    RollbackRequest,
    ShadowRequest,
)
from copper_mvp.release_sources import ArtifactSources


class Retirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: int = Field(ge=1)


def release_router():
    router = APIRouter(prefix="/api/v2")
    lock = threading.Lock()

    def service(request):
        wb = request.app.state.workbench
        with lock:
            if not hasattr(wb, "_model_lifecycle"):
                wb._model_lifecycle = ModelLifecycle(wb.store, wb.access, ArtifactSources(wb.root, wb.data))
        return wb._model_lifecycle

    def actor(request, permission="read"):
        value = request.state.principal
        value.require(permission)
        return value

    @router.get("/model-artifacts")
    def artifacts(request: Request):
        return {"items": service(request).artifacts(actor(request))}

    @router.post("/model-artifacts", status_code=201)
    def register(request: Request, payload: ArtifactRequest):
        return service(request).register(actor(request, "compute"), payload)

    @router.get("/model-artifacts/{identifier}")
    def artifact(request: Request, identifier: str):
        return service(request).artifact(actor(request), identifier)

    @router.post("/model-artifacts/{identifier}/shadow", status_code=202)
    def shadow(request: Request, identifier: str, payload: ShadowRequest):
        return service(request).start_shadow(
            actor(request, "compute"), identifier, payload, request.app.state.workbench.executor
        )

    @router.get("/model-artifacts/{identifier}/shadows")
    def shadows(request: Request, identifier: str):
        return {"items": service(request).shadows(actor(request), identifier)}

    @router.get("/model-shadows/{identifier}")
    def shadow_result(request: Request, identifier: str):
        return service(request).shadow(actor(request), identifier)

    @router.get("/model-artifacts/{identifier}/qualification")
    def qualification(request: Request, identifier: str, shadow_id: str | None = None):
        return service(request).qualification(actor(request), identifier, shadow_id)

    @router.post("/model-artifacts/{identifier}/retire")
    def retire(request: Request, identifier: str, payload: Retirement):
        return service(request).retire(actor(request, "approve"), identifier, payload.expected_version)

    @router.get("/model-pointers")
    def pointers(request: Request):
        return service(request).pointers(actor(request))

    @router.get("/releases")
    def proposals(request: Request):
        return {"items": service(request).proposals(actor(request))}

    @router.post("/releases", status_code=201)
    def propose(request: Request, payload: ReleaseRequest):
        return service(request).propose(actor(request, "compute"), payload)

    @router.get("/releases/{identifier}")
    def proposal(request: Request, identifier: str):
        return service(request).proposal(actor(request), identifier)

    @router.post("/releases/{identifier}/approve")
    def approve(request: Request, identifier: str, payload: ReleaseApproval):
        return service(request).approve(actor(request, "approve"), identifier, payload)

    @router.post("/release-rollbacks", status_code=201)
    def rollback(request: Request, payload: RollbackRequest):
        return service(request).rollback(actor(request, "approve"), payload)

    @router.post("/release-predictions", status_code=201)
    def forecast(request: Request, payload: ForecastRequest):
        return service(request).forecast(actor(request, "compute"), payload)

    @router.get("/release-predictions/{identifier}")
    def prediction(request: Request, identifier: str):
        return service(request).prediction(actor(request), identifier)

    return router
