from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from copper_mvp.common import APP_VERSION, DEFAULT_RUNS_DIR, PROJECT_ROOT, WorkbenchError, dumps, safe
from copper_mvp.contracts import AgentDiagnosticRequest, ExplanationRequest, RunRequest, SelectionRequest
from copper_mvp.data import DataRepository
from copper_mvp.workflows import Workbench
from copper_mvp.api_data import data_router
from copper_mvp.api_models import model_router
from copper_mvp.api_optimizers import optimizer_router
from copper_mvp.api_research import research_router
from copper_mvp.api_control import control_router
from copper_mvp.api_routing import routing_router
from copper_mvp.api_ensembles import ensemble_router
from copper_mvp.api_releases import release_router
from copper_mvp.api_portfolios import portfolio_router
from copper_mvp.api_classical import classical_router
from copper_mvp.api_knowledge import knowledge_router
from copper_mvp.access import Principal, PROJECT


def create_app(run_dir: Path | None = None, data: DataRepository | None = None, *, enable_g1: bool | None = None, enforce_auth: bool = True, frontend_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.workbench = Workbench(run_dir or Path(os.environ.get("COPPER_MVP_RUN_DIR", str(DEFAULT_RUNS_DIR))), data=data)
        yield
        app.state.workbench.close()

    app = FastAPI(title="CuLab 本地研究工作台", version=APP_VERSION, lifespan=lifespan, docs_url=None, redoc_url=None)

    if enable_g1 is None:
        enable_g1 = os.environ.get("COPPER_MVP_G1_ENABLED", "true").lower() not in ("0", "false", "off")
    if enable_g1:
        app.include_router(data_router())

    if os.environ.get("COPPER_MVP_G2A_ENABLED", "true").lower() not in ("0", "false", "off"):
        app.include_router(model_router())

    app.include_router(optimizer_router())

    app.state.enforce_auth = enforce_auth
    app.include_router(research_router())
    app.include_router(control_router())
    app.include_router(routing_router())
    app.include_router(ensemble_router())
    app.include_router(release_router())
    app.include_router(portfolio_router())
    app.include_router(classical_router())
    app.include_router(knowledge_router())

    def workbench(request: Request) -> Workbench:
        return request.app.state.workbench

    @app.exception_handler(WorkbenchError)
    async def domain_error(request, exc):
        status = 401 if exc.code == "UNAUTHENTICATED" else 403 if exc.code == "FORBIDDEN" else 409 if exc.code in ("REQUEST_CONFLICT", "CALL_ALREADY_RESERVED", "SOURCE_CHANGED", "VERSION_CONFLICT", "TASK_STATE", "SESSION_BUSY", "RELEASE_STATE", "ARTIFACT_STATE", "SHADOW_STATE") else 404 if exc.code.endswith("NOT_FOUND") else 400
        return JSONResponse(status_code=status, content={"error": {"code": exc.code, "message": str(exc)}})

    @app.middleware("http")
    async def local_origin(request, call_next):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
            return JSONResponse(status_code=403, content={"error": {"code": "LOCAL_ONLY", "message": "此工作台在本机使用"}})
        if request.url.path.startswith("/api/"):
            if enforce_auth and request.url.hostname not in ("localhost", "127.0.0.1", "::1"):
                return JSONResponse(status_code=403, content={"error": {"code": "LOCAL_ONLY", "message": "请从本机地址访问"}})
            authorization = request.headers.get("authorization", "")
            key = authorization[7:] if authorization.startswith("Bearer ") else None
            principal = request.app.state.workbench.access.authenticate(key=key, cookie=request.cookies.get("culab_session"))
            if not enforce_auth and principal is None:
                principal = Principal("owner", "owner")
            request.state.principal = principal
            if request.url.path not in ("/api/auth/status", "/api/auth/session"):
                if principal is None:
                    return JSONResponse(status_code=401, content={"error": {"code": "UNAUTHENTICATED", "message": "需要本机访问码"}})
                if principal.project_id != PROJECT:
                    return JSONResponse(status_code=403, content={"error": {"code": "FORBIDDEN", "message": "当前账号无权访问此项目"}})
                legacy_write = request.method != "GET" and request.url.path.startswith("/api/runs")
                compute_write = request.method == "POST" and request.url.path == "/api/v2/model-comparisons"
                if legacy_write or compute_write:
                    try:
                        principal.require("compute")
                    except WorkbenchError:
                        return JSONResponse(status_code=403, content={"error": {"code": "FORBIDDEN", "message": "当前账号没有计算权限"}})
        response = await call_next(request)
        if request.url.path.startswith("/api/v2/knowledge/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/health")
    def health(request: Request):
        wb = workbench(request)
        return {"status": "ready", "version": APP_VERSION, "local_only": True, "llm_enabled": wb.explanations.enabled, "models_ready": bool(wb.models.catalog()), "dataset_version": wb.data.dataset_version}

    @app.get("/api/overview")
    def overview(request: Request):
        wb = workbench(request)
        return {**wb.data.overview(), "models": wb.models.catalog(), "recent_runs": wb.store.list(limit=6)}

    @app.get("/api/diagnostic-agent")
    def diagnostic_agent(request: Request):
        return workbench(request).diagnostic_agent.availability()

    @app.post("/api/runs/{run_id}/diagnoses", status_code=202)
    def create_diagnosis(request: Request, run_id: str, payload: AgentDiagnosticRequest):
        return workbench(request).submit_diagnosis(run_id, payload)

    @app.get("/api/runs/{run_id}/diagnoses")
    def diagnoses(request: Request, run_id: str):
        return {"items": workbench(request).store.diagnoses(run_id)}

    @app.get("/api/events")
    def events(request: Request, year: int | None = None, query: str = "", scope: Literal["all", "oof", "dual"] = "all", offset: int = Query(0, ge=0), limit: int = Query(40, ge=1, le=100)):
        return workbench(request).data.events(year, query[:120], scope, offset, limit)

    @app.get("/api/events/{event_id}")
    def event(request: Request, event_id: str):
        return workbench(request).data.context(event_id)

    @app.get("/api/models")
    def models(request: Request):
        return {"items": workbench(request).models.catalog(), "presets": ["Persistence", "DeltaRidge", "DeltaHGB"]}

    @app.get("/api/models/{bundle_id}/series")
    def model_series(request: Request, bundle_id: str, target: Literal["cu", "as"] = "cu"):
        return {"items": workbench(request).models.series(bundle_id, target)}

    @app.get("/api/experiments")
    def experiments(request: Request):
        history = []
        for name, subfolder in (("P2 历史基线", "baselines_v1"), ("P2.1 历史变化量", "residual_baselines_v1")):
            path = workbench(request).data.evidence_dir / "artifacts/p2" / subfolder / "overall_metrics_v1.csv"
            if path.is_file():
                rows = pd.read_csv(path)
                history.extend({"experiment": name, "model": r.model_name, "target": "cu" if r.target_name == "target_cu_g_l" else "as", "n": r.validation_sample_count, "mae": r.pooled_mae, "rmse": r.pooled_rmse, "r2": r.pooled_r2, "run_id": r.run_id, "source": str(path.relative_to(workbench(request).data.evidence_dir))} for r in rows.itertuples())
        return safe({"history": history, "bundles": workbench(request).models.catalog()})

    @app.post("/api/runs", status_code=202)
    def create_run(request: Request, payload: RunRequest):
        return workbench(request).submit(payload)

    @app.get("/api/runs")
    def runs(request: Request, task_type: str | None = None, limit: int = Query(100, ge=1, le=200)):
        return {"items": workbench(request).store.list(task_type, limit)}

    @app.get("/api/runs/{run_id}")
    def run(request: Request, run_id: str):
        return workbench(request).store.get(run_id)

    @app.post("/api/runs/{run_id}/selection")
    def selection(request: Request, run_id: str, payload: SelectionRequest):
        wb = workbench(request)
        wb.store.select(run_id, payload.candidate_id, payload.label)
        return wb.store.get(run_id)

    @app.post("/api/runs/{run_id}/explanation")
    def explanation(request: Request, run_id: str, payload: ExplanationRequest):
        return workbench(request).explanations.explain(run_id, payload.question, payload.use_llm)

    @app.get("/api/runs/{run_id}/export")
    def export(request: Request, run_id: str, format: Literal["json", "csv", "md"] = "md"):
        wb = workbench(request)
        item = wb.store.get(run_id)
        result = item.get("result") or {}
        headers = {"Content-Disposition": f'attachment; filename="copper-{run_id}.{format}"'}
        if format == "json":
            return Response(dumps(item), media_type="application/json", headers=headers)
        if format == "csv":
            path = wb.store.directory(run_id) / "pareto_candidates.csv"
            if result.get("kind") == "optimization" and path.is_file():
                return FileResponse(path, media_type="text/csv", headers=headers)
            if result.get("kind") == "training":
                rows = result.get("metrics", [])
            elif result.get("kind") == "prediction":
                rows = [{"target": k, **v} for k, v in result.get("predictions", {}).items()]
            elif result.get("kind") == "agent_diagnosis":
                rows = result.get("steps", [])
            else:
                rows = [{"run_id": run_id, "task_type": item["task_type"], "status": item["status"], "solution_status": result.get("solution_status"), "code": result.get("code") or (item.get("error") or {}).get("code"), "message": result.get("message") or (item.get("error") or {}).get("message")}]
            return Response(pd.DataFrame(rows).to_csv(index=False), media_type="text/csv; charset=utf-8", headers=headers)
        from copper_mvp.explanation import template
        text = f"# CuLab 运行摘要\n\n运行编号：{run_id}\n\n任务：{item['task_type']}\n\n状态：{item['status']}\n\n"
        if result.get("kind") == "agent_diagnosis":
            report = result.get("report") or {}
            text += report.get("summary", (item.get("error") or {}).get("message", "诊断进行中"))
            for finding in report.get("findings", []):
                text += "\n\n- " + finding["claim"] + " [" + ", ".join(finding["evidence_ids"]) + "]"
        elif item["status"] == "completed":
            text += template(item, "summary")["text"]
        elif item["error"]:
            text += item["error"]["message"]
        text += "\n\n## 计算结果\n\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n"
        return Response(text, media_type="text/markdown; charset=utf-8", headers=headers)

    dist = frontend_dir or PROJECT_ROOT / "web/dist"
    app.mount("/assets", StaticFiles(directory=dist / "assets", check_dir=False), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api/"):
            return JSONResponse(status_code=404, content={"error": {"code": "NOT_FOUND", "message": "没有该接口"}})
        index = dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        return JSONResponse({"message": "后端已就绪，请构建 web 前端后刷新。", "api_schema": "/openapi.json"})

    return app
