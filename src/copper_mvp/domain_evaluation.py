"""Small prompt/RAG studies built on the existing research-task service."""

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from copper_mvp.common import WorkbenchError, digest, utc_now, write_json
from copper_mvp.knowledge_contracts import DocumentSpec
from copper_mvp.model_comparisons import observed_job_state
from copper_mvp.research_documents import DisclosureConsent
from copper_mvp.research_service import PROMPT

DEFAULT_QUESTIONS = [
    "当前项目中 Cu 与 As 分别使用什么单位，比较结果应该如何标注？",
    "合成 Mock 已收到命令 ACK，是否可以直接判定调节成功？怎样确认并处理 ACK 丢失？",
]
DEFAULT_DOCUMENT = (
    "# 合成项目规则\n"
    "本资料用于 CuLab 的领域评测，不表示真实工厂设定。Cu 浓度使用 g/L，"
    "As 浓度使用 mg/L，展示或比较时应分别标注单位。合成 Mock 的命令 ACK"
    " 表示设备已接收命令；达到目标还需要三个连续合格回读。ACK 丢失时先"
    "查询设备账本和反馈，同一命令不直接重复下发。"
)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    title: str = Field(default="项目规则对照", min_length=1, max_length=120)
    questions: list[str] = Field(default_factory=lambda: list(DEFAULT_QUESTIONS), min_length=1, max_length=4)
    document_text: str = Field(default=DEFAULT_DOCUMENT, min_length=20, max_length=6000)
    source_note: str = Field(default="合成项目规则，用于工程评测", min_length=1, max_length=300)
    max_cost_cny: float = Field(default=2, gt=0, le=2)


class EvaluationReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["accepted", "needs_revision"]
    notes: str = Field(min_length=1, max_length=2000)


class DomainEvaluationService:
    def __init__(self, workbench):
        self.wb = workbench
        self.root = workbench.root / "domain_evaluations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.futures = {}

    def close(self):
        self.stop.set()

    def defaults(self):
        return {
            "questions": DEFAULT_QUESTIONS,
            "document_text": DEFAULT_DOCUMENT,
            "variants": ["prompt", "rag"],
            "max_cost_cny": 2,
            "max_output_tokens": 2048,
            "can_run": self.wb.research.allow_live or self.wb.research.transport is not None,
            "provider_mode": "test_transport" if self.wb.research.transport is not None else "live",
            "weight_training": "not_started",
        }

    def path(self, identifier):
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise WorkbenchError("评测不存在", "DOMAIN_NOT_FOUND")
        return self.root / identifier / "state.json"

    def save(self, state):
        with self.lock:
            path = self.path(state["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            write_json(temporary, state)
            temporary.replace(path)

    def read(self, actor, identifier):
        actor.require("read")
        with self.lock:
            path = self.path(identifier)
            if not path.exists():
                raise WorkbenchError("评测不存在", "DOMAIN_NOT_FOUND")
            state = json.loads(path.read_text(encoding="utf-8"))
        if state["owner_id"] != actor.user_id or state["project_id"] != actor.project_id:
            raise WorkbenchError("评测不存在", "DOMAIN_NOT_FOUND")
        return observed_job_state(state)

    def list(self, actor):
        actor.require("read")
        result = []
        for path in sorted(self.root.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw["owner_id"] == actor.user_id and raw["project_id"] == actor.project_id:
                result.append(observed_job_state(raw))
        return result

    def get(self, actor, identifier):
        state = self.read(actor, identifier)
        results = []
        for row in state["cases"]:
            item = dict(row)
            if row.get("task_id"):
                task = self.wb.research.present_task(actor, row["task_id"])
                item.update(
                    task_status=task["status"],
                    result=task.get("result"),
                    usage=task.get("usage"),
                    error=task.get("error"),
                )
            results.append(item)
        summaries = []
        for variant in ("prompt", "rag"):
            rows = [row for row in state["cases"] if row["variant"] == variant]
            times = [row["elapsed_ms"] for row in rows if row.get("elapsed_ms") is not None]
            summaries.append(
                {
                    "variant": variant,
                    "n": len(rows),
                    "completed": sum(row["status"] == "completed" for row in rows),
                    "answers": sum(row.get("answer_kind") == "answer" for row in rows),
                    "document_citations": sum(row.get("document_citations", 0) for row in rows),
                    "cost_cny": sum(row.get("usage", {}).get("spent_cny", 0) for row in rows),
                    "unsettled_cny": sum(row.get("usage", {}).get("unsettled_cny", 0) for row in rows),
                    "p50_ms": float(np.percentile(times, 50)) if times else None,
                    "p95_ms": float(np.percentile(times, 95)) if times else None,
                }
            )
        return {**state, "cases": results, "summary": summaries}

    def create(self, actor, request):
        actor.require("manage")
        request = EvaluationRequest.model_validate(request)
        if any(not question.strip() or len(question) > 1000 for question in request.questions):
            raise WorkbenchError("问题不能为空，每题最多1000字", "DOMAIN_INPUT")
        identifier = digest([actor.project_id, actor.user_id, request.request_key])[:32]
        fingerprint = digest(request.model_dump())
        with self.lock:
            if self.path(identifier).exists():
                old = self.read(actor, identifier)
                if old["fingerprint"] != fingerprint:
                    raise WorkbenchError("请求键对应的方案已经保存", "REQUEST_CONFLICT")
                return old
            doc_id = "domain-" + identifier
            try:
                document = self.wb.knowledge.inspect(actor, doc_id, 1)
            except WorkbenchError as exc:
                if not exc.code.endswith("NOT_FOUND"):
                    raise
                self.wb.knowledge.register(
                    actor,
                    DocumentSpec(
                        doc_id=doc_id,
                        title=request.title + " · 评测资料",
                        author=actor.user_id,
                        source=request.source_note,
                        authorization="用于当前用户创建的领域对照评测",
                        license="由资料提交者确认其使用范围",
                        version_label="v1",
                        format="md",
                        effective_at=datetime.now(UTC),
                    ),
                )
                document = None
            uploaded = self.wb.knowledge.upload(actor, doc_id, 1, request.document_text.encode())
            if uploaded["status"] != "indexed":
                self.wb.knowledge.review(actor, doc_id, 1, True, "评测方案提交的文本资料")
            self.wb.knowledge.build_index(actor, doc_id, 1)
            document = self.wb.knowledge.inspect(actor, doc_id, 1)
            state = {
                "id": identifier,
                "owner_id": actor.user_id,
                "project_id": actor.project_id,
                "created_at": utc_now(),
                "title": request.title,
                "fingerprint": fingerprint,
                "source_note": request.source_note,
                "status": "planned",
                "questions": request.questions,
                "document": {
                    "doc_id": doc_id,
                    "version": 1,
                    "content_hash": document["content_hash"],
                    "parse_hash": document["index_parse_hash"],
                },
                "max_cost_cny": request.max_cost_cny,
                "prompt_hash": digest(PROMPT),
                "source_version": self.wb.research.source_version(),
                "cases": [
                    {"case_id": str(i + 1), "question": question, "variant": variant, "status": "pending"}
                    for variant in ("prompt", "rag")
                    for i, question in enumerate(request.questions)
                ],
                "review": None,
            }
            self.save(state)
            return state

    def run(self, actor, identifier):
        actor.require("approve")
        with self.lock:
            state = self.read(actor, identifier)
            if state["status"] == "completed":
                return state
            if identifier in self.futures and not self.futures[identifier].done():
                return state
            if not self.defaults()["can_run"]:
                raise WorkbenchError("当前研究助手未启用真实调用", "DOMAIN_PROVIDER_DISABLED")
            if state["source_version"] != self.wb.research.source_version():
                raise WorkbenchError("研究配置已变化，请保存新的评测方案", "SOURCE_CHANGED")
            doc = state["document"]
            self.wb.research.documents.grant(
                actor,
                doc["doc_id"],
                1,
                DisclosureConsent(
                    document_hash=doc["content_hash"],
                    parse_hash=doc["parse_hash"],
                    expires_at=datetime.now(UTC) + timedelta(days=1),
                    reason="运行领域评测：" + state["title"],
                    max_excerpt_tokens=1200,
                ),
            )
            state.pop("error", None)
            state.update(status="queued", owner_pid=os.getpid(), provider_mode=self.defaults()["provider_mode"])
            self.save(state)
            self.futures[identifier] = self.wb.executor.submit(self.execute, actor, identifier)
            return state

    def execute(self, actor, identifier):
        state = self.read(actor, identifier)
        state["status"] = "running"
        self.save(state)
        try:
            cap = min(0.5, state["max_cost_cny"] / len(state["cases"]))
            for index, row in enumerate(state["cases"]):
                if self.stop.is_set():
                    state["status"] = "interrupted"
                    break
                if row["status"] in ("completed", "failed", "cancelled", "needs_attention") and row.get("task_id"):
                    continue
                context = {}
                if row["variant"] == "rag":
                    found = self.wb.knowledge.search(actor, row["question"], limit=5)
                    matching = [v for v in found["items"] if v["citation"]["doc_id"] == state["document"]["doc_id"]]
                    if not matching:
                        raise WorkbenchError("评测资料未检索到匹配片段", "DOMAIN_DOCUMENT")
                    citation = matching[0]["citation"]
                    context = {
                        "knowledge_refs": [{k: citation[k] for k in ("chunk_id", "hash", "parse_hash", "as_of")}]
                    }
                if not row.get("session_id"):
                    row["session_id"] = self.wb.research.store.create_session(
                        actor, state["title"] + " · " + row["variant"] + " · " + row["case_id"], context
                    )["id"]
                    self.save(state)
                if not row.get("task_id"):
                    task = self.wb.research.submit(
                        actor,
                        row["session_id"],
                        row["question"],
                        "domain:" + identifier + ":" + str(index),
                        context,
                        max_cost_cny=cap,
                        experiment_profile=row["variant"],
                    )
                    row["task_id"] = task["id"]
                    self.save(state)
                existing = self.wb.research.present_task(actor, row["task_id"])
                if existing["status"] == "paused":
                    self.wb.research.store.control(
                        actor, row["task_id"], "resume", existing["version"], self.wb.research.source_version()
                    )
                    self.wb.research.schedule(row["task_id"])
                while not self.stop.is_set():
                    task = self.wb.research.present_task(actor, row["task_id"])
                    if task["status"] not in ("queued", "running", "pausing", "cancelling"):
                        break
                    self.stop.wait(0.2)
                if self.stop.is_set():
                    state["status"] = "interrupted"
                    break
                row.update(
                    status=task["status"],
                    elapsed_ms=task.get("elapsed_ms"),
                    model_calls=task["model_calls"],
                    tool_calls=task["tool_calls"],
                    usage=task["usage"],
                    answer_kind=(task.get("result") or {}).get("kind"),
                    evidence_count=len((task.get("result") or {}).get("evidence_ids", [])),
                    document_citations=len((task.get("result") or {}).get("document_refs", [])),
                )
                self.save(state)
                if task["status"] == "paused":
                    state["status"] = "interrupted"
                    break
            else:
                state["status"] = "completed"
            state["finished_at"] = utc_now()
        except Exception as exc:
            state.update(
                status="failed", error={"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)[:300]}
            )
        finally:
            self.save(state)

    def review(self, actor, identifier, request):
        actor.require("compute")
        state = self.read(actor, identifier)
        if state["status"] != "completed":
            raise WorkbenchError("完成评测后再填写复核", "TASK_STATE")
        state["review"] = {**request.model_dump(), "reviewed_by": actor.user_id, "reviewed_at": utc_now()}
        self.save(state)
        return self.get(actor, identifier)
