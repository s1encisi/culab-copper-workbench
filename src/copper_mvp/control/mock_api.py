"""Loopback-only Mock server with separate driver and test-admin credentials."""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from copper_mvp.common import WorkbenchError
from copper_mvp.control.contracts import NOTICE, DispatchInput, FaultInput, TickInput
from copper_mvp.control.mock import MockDevice


def local_key(path):
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    return path.read_text(encoding="utf-8").strip()


def create_mock_app(root: Path, *, manual_clock=False, seed=17):
    device = MockDevice(root, seed)
    driver_key = local_key(device.root / "driver.key")
    admin_key = local_key(device.root / "test_admin.key")
    stopped = threading.Event()

    def advance():
        previous = time.monotonic()
        while not stopped.wait(0.2):
            now = time.monotonic()
            device.tick(min(60, now - previous))
            previous = now

    @asynccontextmanager
    async def lifespan(app):
        device.inject_fault({"restart": True})
        thread = None
        if not manual_clock:
            thread = threading.Thread(target=advance, name="mock-clock", daemon=True)
            thread.start()
        yield
        stopped.set()
        if thread:
            thread.join(timeout=2)

    app = FastAPI(title=NOTICE, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.device = device

    @app.middleware("http")
    async def authorize(request, call_next):
        if request.url.hostname not in ("localhost", "127.0.0.1", "::1") or request.headers.get("origin"):
            return JSONResponse(status_code=403, content={"error": {"code": "MOCK_LOCAL_ONLY"}})
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        required = admin_key if request.url.path.startswith("/v1/test-admin/") else driver_key
        if not hmac.compare_digest(token, required):
            return JSONResponse(status_code=401, content={"error": {"code": "UNAUTHENTICATED"}})
        return await call_next(request)

    @app.exception_handler(WorkbenchError)
    async def error(request, exc):
        return JSONResponse(status_code=409, content={"error": {"code": exc.code, "message": str(exc)}})

    @app.get("/v1/state")
    def state():
        return device.read_state()

    @app.get("/v1/points")
    def points():
        return device.read_points()

    @app.post("/v1/validate")
    def validate(command: dict):
        return device.validate_command(command)

    @app.post("/v1/commands")
    def submit(payload: DispatchInput):
        receipt = device.submit_command(**payload.model_dump())
        if receipt["ack_lost"]:
            return JSONResponse(
                status_code=504, content={"error": {"code": "UNKNOWN_OUTCOME", "message": "合成 ACK 丢失"}}
            )
        receipt["ack_received"] = device.read_state()["virtual_time"] >= receipt["ack_at"]
        return receipt

    @app.get("/v1/commands/{command_id}")
    def command(command_id: str):
        return {"record": device.get_command_status(command_id)}

    @app.post("/v1/test-admin/tick")
    def tick(payload: TickInput):
        return device.tick(payload.seconds)

    @app.post("/v1/test-admin/faults")
    def faults(payload: FaultInput):
        return device.inject_fault(payload.model_dump(exclude_none=True))

    return app
