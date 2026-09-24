"""Authenticated sessions, messages and authoritative task controls."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Literal

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr

from copper_mvp.common import WorkbenchError
from copper_mvp.research_memory import KnowledgeReference, MemoryCapture
from copper_mvp.research_store import TERMINAL


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optimization_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    model_comparison_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    optimizer_comparison_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    event_id: str | None = Field(default=None, max_length=180)
    as_of: AwareDatetime | None = None
    knowledge_refs: list[KnowledgeReference] = Field(default_factory=list, max_length=20)
    control_command_ids: list[Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]] = Field(
        default_factory=list, max_length=20
    )


class NewSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="研究对话", min_length=1, max_length=80)
    context: Context = Field(default_factory=Context)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=6000)
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    context: Context | None = None
    max_cost_cny: float | None = Field(default=None, gt=0, le=2)


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: int = Field(ge=1)


class Login(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_code: SecretStr


class NewAccess(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=80)
    role: Literal["owner", "researcher", "viewer"]


def research_router():
    router = APIRouter()

    def wb(request):
        return request.app.state.workbench

    def actor(request):
        principal = getattr(request.state, "principal", None)
        if principal is None:
            raise WorkbenchError("需要本机访问码", "UNAUTHENTICATED")
        return principal

    @router.get("/api/auth/status")
    def status(request: Request):
        value = getattr(request.state, "principal", None)
        return {
            "authenticated": value is not None,
            "user": value.user_id if value else None,
            "role": value.role if value else None,
            "access_code_location": "运行目录中的 owner_access.key",
            "auth_required": request.app.state.enforce_auth,
        }

    @router.post("/api/auth/session")
    def login(request: Request, response: Response, payload: Login):
        principal, cookie = wb(request).access.login(payload.access_code.get_secret_value())
        response.set_cookie(
            "culab_session",
            cookie,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=30 * 86400,
        )
        return {"user": principal.user_id, "role": principal.role}

    @router.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        wb(request).access.logout(request.cookies.get("culab_session"))
        response.delete_cookie("culab_session")
        return {"logged_out": True}

    @router.post("/api/v2/access-keys")
    def issue_key(request: Request, payload: NewAccess):
        key = wb(request).access.issue(actor(request), payload.user_id, payload.role)
        return {"access_code": key, "user_id": payload.user_id, "role": payload.role}

    @router.get("/api/v2/assistant")
    def assistant_info(request: Request):
        actor(request).require("read")
        service = wb(request).research
        return {
            "model": service.settings.model,
            "live_calls_enabled": service.allow_live,
            "max_calls": service.settings.max_calls,
            "max_tools": service.settings.max_tool_calls,
            "max_cost_cny": service.settings.max_cost_cny,
            "session_cost_cny": 10,
            "user_day_cost_cny": 20,
        }

    @router.post("/api/v2/sessions", status_code=201)
    def new_session(request: Request, payload: NewSession):
        return wb(request).research.store.create_session(
            actor(request), payload.title, payload.context.model_dump(mode="json")
        )

    @router.get("/api/v2/sessions")
    def sessions(request: Request):
        return {"items": wb(request).research.store.sessions(actor(request))}

    @router.get("/api/v2/sessions/{session_id}")
    def session(request: Request, session_id: str):
        return wb(request).research.present_session(actor(request), session_id)

    @router.post("/api/v2/sessions/{session_id}/memory")
    def capture_memory(request: Request, session_id: str, payload: MemoryCapture):
        service = wb(request).research
        return service.memory.capture(actor(request), session_id, payload.ttl_hours, service.source_version())

    @router.get("/api/v2/sessions/{session_id}/memory")
    def memory(request: Request, session_id: str):
        service = wb(request).research
        return service.memory.restore(actor(request), session_id, service.source_version())

    @router.delete("/api/v2/sessions/{session_id}/memory")
    def forget_memory(request: Request, session_id: str):
        return wb(request).research.memory.forget(actor(request), session_id)

    @router.post("/api/v2/sessions/{session_id}/messages", status_code=202)
    def message(request: Request, session_id: str, payload: Message):
        return wb(request).research.submit(
            actor(request),
            session_id,
            payload.question,
            payload.request_key,
            payload.context.model_dump(mode="json", exclude_unset=True) if payload.context else None,
            payload.max_cost_cny,
        )

    @router.get("/api/v2/tasks/{task_id}")
    def task(request: Request, task_id: str):
        return wb(request).research.present_task(actor(request), task_id)

    @router.post("/api/v2/tasks/{task_id}/{action}")
    def control(request: Request, task_id: str, action: Literal["pause", "resume", "cancel"], payload: Control):
        service = wb(request).research
        result = service.store.control(
            actor(request),
            task_id,
            action,
            payload.expected_version,
            service.source_version() if action == "resume" else "",
        )
        if action == "resume":
            service.schedule(task_id)
        return result

    @router.get("/api/v2/tasks/{task_id}/events")
    def events(request: Request, task_id: str, after: int = 0):
        principal = actor(request)
        store = wb(request).research.store
        store.task(principal, task_id)

        async def stream():
            sequence = max(0, after)
            while not await request.is_disconnected():
                for item in store.events(principal, task_id, sequence):
                    sequence = item["seq"]
                    yield "id: " + str(sequence) + "\ndata: " + json.dumps(item, ensure_ascii=False) + "\n\n"
                current = store.internal_task(task_id)
                if current["status"] in TERMINAL or current["status"] == "paused":
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return router
