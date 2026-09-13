from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from copper_mas.config.settings import redact_text
from copper_mvp.checks import verify_prediction
from copper_mvp.common import DEFAULT_RUNS_DIR, WorkbenchError, digest, utc_now, write_json
from copper_mvp.contracts import AgentDiagnosticRequest, RunRequest
from copper_mvp.data import DataRepository
from copper_mvp.diagnostic_agent import AGENT_VERSION, DiagnosticAgentService
from copper_mvp.explanation import ExplanationService
from copper_mvp.modeling import MODEL_VERSION, ModelManager
from copper_mvp.optimization import solve
from copper_mvp.storage import RunStore
from copper_mvp.research_store import ResearchStore
from copper_mvp.knowledge_store import KnowledgeStore
from copper_mvp.knowledge_reranker import LocalReranker
from copper_mvp.calibration_service import CalibrationService
from copper_mvp.access import AccessControl
from copper_mvp.research_service import ResearchService
from copper_mvp.control.client import MockClient
from copper_mvp.control.commands import CommandService


class GraphState(TypedDict, total=False):
    run_id: str
    request: dict
    result_ref: str
    event_ref: str


class RuntimeLock:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / "runtime.lock").open("a+b")
        self.file.seek(0)
        if self.file.read(1) == b"":
            self.file.write(b"0"); self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("该运行目录已有一个工作台后台，请使用已有服务或另选运行目录") from exc

    def close(self):
        self.file.close()


class Workbench:
    def __init__(self, root: Path = DEFAULT_RUNS_DIR, data: DataRepository | None = None):
        self.root = Path(root)
        self.owner = RuntimeLock(self.root)
        self.store = RunStore(self.root)
        self.store.recover_interrupted()
        self.data = data or DataRepository()
        self.models = ModelManager(self.root, self.data)
        self.explanations = ExplanationService(self.store)
        self.diagnostic_agent = DiagnosticAgentService(self.store, self.data, self.models)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="copper-task")
        self.futures = {}
        self.lock = threading.Lock()
        self.access = AccessControl(ResearchStore(self.store))
        self.knowledge = KnowledgeStore(self.root, reranker=LocalReranker())
        self.research = ResearchService(self, self.access)
        self.control = CommandService(self.store, self.access, MockClient.from_env())
        self.calibrations = CalibrationService(self.root, self.data)

    def close(self):
        self.control.close()
        self.research.close()
        self.executor.shutdown(wait=True)
        self.owner.close()

    def submit(self, request: RunRequest) -> dict:
        payload = request.model_dump()
        if request.event_id:
            self.data.row(request.event_id)
        effective_bundle = request.bundle_id
        uses_response = request.task_type == "predict" or (request.task_type == "optimize" and request.mode == "plant")
        if uses_response and request.model_profile != "Persistence":
            try:
                effective_bundle = self.models.manifest(request.bundle_id)["bundle_id"]
            except WorkbenchError as exc:
                if request.task_type == "optimize" or request.model_profile != "auto" or request.bundle_id is not None or exc.code != "MODEL_NOT_READY":
                    raise
                payload["model_profile"] = "Persistence"
        payload["bundle_id"] = effective_bundle
        fingerprint = digest({"request": {k: v for k, v in payload.items() if k != "request_key"}, "dataset": self.data.dataset_version, "models": MODEL_VERSION, "graph": "mvp-graph-v1"})
        run, reused = self.store.create(payload, fingerprint)
        if not reused:
            self.futures[run["run_id"]] = self.executor.submit(self._execute, run["run_id"], payload)
        return {**run, "reused": reused}

    def submit_diagnosis(self, source_run_id: str, request: AgentDiagnosticRequest) -> dict:
        source = self.store.get(source_run_id)
        result = source.get("result") or {}
        if source["status"] != "completed" or result.get("kind") != "optimization" or result.get("mode") != "plant":
            raise WorkbenchError("请选择已完成的工厂优化运行", "DIAGNOSIS_SOURCE")
        fingerprint = digest({"source": result, "source_run_id": source_run_id, "question": request.question, "agent": AGENT_VERSION, "settings": self.diagnostic_agent.settings.model_dump()})
        payload = {"task_type": "agent_diagnostic", "source_run_id": source_run_id, "question": request.question, "request_key": request.request_key or f"diag:{source_run_id}:{fingerprint[:20]}", "model": self.diagnostic_agent.settings.model}
        run, reused = self.store.create(payload, fingerprint)
        if not reused:
            self.futures[run["run_id"]] = self.executor.submit(self.diagnostic_agent.execute, run["run_id"])
        return {**run, "reused": reused}

    def _execute(self, run_id: str, request: dict):
        start = perf_counter()
        self.store.start(run_id)
        try:
            with SqliteSaver.from_conn_string(str(self.root / "checkpoints.sqlite")) as saver:
                graph = self._graph(saver)
                state = graph.invoke({"run_id": run_id, "request": request}, {"configurable": {"thread_id": run_id}})
            result = json.loads(Path(state["result_ref"]).read_text(encoding="utf-8"))
            self.store.finish(run_id, result, (perf_counter() - start) * 1000)
        except Exception as exc:
            self.store.fail(run_id, getattr(exc, "code", type(exc).__name__), redact_text(str(exc)), (perf_counter() - start) * 1000)

    def _graph(self, saver):
        def progress(run_id):
            return lambda node, detail, duration: self.store.trace(run_id, node, "completed", utc_now(), duration, detail)

        def wrap(name, fn):
            def node(state):
                started = utc_now(); clock = perf_counter()
                try:
                    result = fn(state)
                    self.store.trace(state["run_id"], name, "completed", started, (perf_counter() - clock) * 1000, {"result": "完成"})
                    return result
                except Exception as exc:
                    self.store.trace(state["run_id"], name, "failed", started, (perf_counter() - clock) * 1000, {"code": getattr(exc, "code", type(exc).__name__), "message": redact_text(str(exc))})
                    raise
            return node

        def save_draft(state, result):
            path = self.store.directory(state["run_id"]) / "draft_result.json"
            write_json(path, result)
            return {"result_ref": str(path)}

        def admit(state):
            event = state["request"]["event_id"]
            self.data.context(event)
            return {"event_ref": event}

        def mode(state):
            self.data.mode_cards[state["event_ref"]]
            return {}

        def predict(state):
            r = state["request"]
            result = self.models.predict(r["event_id"], r["model_profile"], r["model_scope"], r["bundle_id"])
            result.pop("audit", None)
            return save_draft(state, result)

        def audit(state):
            path = Path(state["result_ref"])
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["kind"] == "prediction":
                result["audit"] = verify_prediction(result, self.data.context(state["event_ref"]))
            elif result["kind"] == "optimization" and not result["audit"]["passed"]:
                raise WorkbenchError("优化复核未通过", "CANDIDATE_AUDIT")
            write_json(path, result)
            return {}

        def train(state):
            result = self.models.train(state["run_id"], progress(state["run_id"]))
            return save_draft(state, {"kind": "training", **result})

        def optimize(state):
            result = solve(self.data, self.models, RunRequest.model_validate(state["request"]), self.store.directory(state["run_id"]), progress(state["run_id"]))
            return save_draft(state, result)

        def diagnostic(state):
            scenario = state["request"]["scenario"]
            event_id = self.data.summaries[-1]["event_id"]
            context = self.data.context(event_id)
            result = self.models.predict(event_id, "Persistence")
            code = "CLEAN_PASSED"
            blocked = False
            try:
                if scenario == "future_field":
                    RunRequest.model_validate({"task_type": "predict", "event_id": event_id, "request_key": "diagnostic", "target_cu_g_l": 1.0})
                elif scenario == "model_mismatch":
                    result["dataset_version"] = "mismatched-version"
                elif scenario == "invalid_numeric":
                    result["predictions"]["cu"]["value"] = float("nan")
                elif scenario == "invalid_interval":
                    p = result["predictions"]["cu"]
                    p["interval"] = [p["value"] + 1, p["value"] + 2]
                verify_prediction(result, context)
            except (WorkbenchError, ValidationError) as exc:
                blocked = True
                code = getattr(exc, "code", "REQUEST_SCHEMA_EXTRA_FIELD")
            expected = scenario != "clean"
            return save_draft(state, {"kind": "diagnostic", "scenario": scenario, "data_kind": "IN_MEMORY_FAULT_COPY", "blocked": blocked, "expected_blocked": expected, "passed": blocked == expected, "code": code, "message": "正常输入通过" if not blocked else "异常被实际请求合同或结果检查拦截，原始数据保持不变。"})

        graph = StateGraph(GraphState)
        graph.add_node("A1", wrap("A1 数据装配", admit))
        graph.add_node("A2", wrap("A2 工况", mode))
        graph.add_node("A4", wrap("A4 数值预测", predict))
        graph.add_node("A5", wrap("A5 结果复核", audit))
        graph.add_node("TRAIN", wrap("A3 模型实验", train))
        graph.add_node("OPTIMIZE", wrap("多目标候选搜索", optimize))
        graph.add_node("DIAGNOSTIC", wrap("异常案例验证", diagnostic))
        def entry(state):
            r = state["request"]
            if r["task_type"] == "train": return "TRAIN"
            if r["task_type"] == "diagnostic": return "DIAGNOSTIC"
            if r["task_type"] == "optimize" and r["mode"] == "benchmark": return "OPTIMIZE"
            return "A1"
        graph.add_conditional_edges(START, entry)
        graph.add_edge("A1", "A2")
        graph.add_conditional_edges("A2", lambda s: "OPTIMIZE" if s["request"]["task_type"] == "optimize" else "A4")
        graph.add_edge("A4", "A5")
        graph.add_edge("OPTIMIZE", "A5")
        graph.add_edge("A5", END); graph.add_edge("TRAIN", END); graph.add_edge("DIAGNOSTIC", END)
        return graph.compile(checkpointer=saver)
