"""Additive, local-only G1 read APIs. No data paths or row uploads are accepted."""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from copper_mvp.data_contracts import LabelQuery, ReplayQuery
from copper_mvp.data_service import DataService


def data_router() -> APIRouter:
    router = APIRouter(prefix="/api/v2/data", tags=["G1 data evidence"])
    lock = threading.Lock()

    def service(request: Request) -> DataService:
        wb = request.app.state.workbench
        with lock:
            if not hasattr(wb, "_g1_data_service"):
                wb._g1_data_service = DataService(wb.data, wb.models)
            return wb._g1_data_service

    def query(request: Request, model=None):
        values = dict(request.query_params)
        if len(request.query_params.multi_items()) != len(values):
            raise HTTPException(422, detail="不允许重复查询参数")
        if model is None:
            if values:
                raise HTTPException(422, detail="此接口不接受额外查询参数")
            return None
        try:
            return model.model_validate(values)
        except ValidationError as exc:
            raise HTTPException(
                422, detail=exc.errors(include_input=False, include_context=False, include_url=False)
            ) from exc

    @router.get("/dependencies")
    def dependencies(request: Request):
        query(request)
        return service(request).dependencies()

    @router.get("/events/{event_id}/task-spec")
    def task_spec(request: Request, event_id: str):
        query(request)
        current = service(request)
        current._verify_sources()
        return current.task_spec(event_id).model_dump(mode="json")

    @router.get("/events/{event_id}/snapshot")
    def snapshot(request: Request, event_id: str):
        query(request)
        return service(request).snapshot(event_id)

    @router.get("/labels")
    def labels(request: Request):
        payload = query(request, LabelQuery)
        current = service(request)
        current._verify_sources()
        return current.labels.snapshot(payload.as_of, days=payload.days, limit=payload.limit, minimum=payload.minimum)

    @router.get("/events/{event_id}/evidence")
    def evidence(request: Request, event_id: str):
        payload = query(request, ReplayQuery)
        return service(request).evidence(event_id, payload)

    return router
