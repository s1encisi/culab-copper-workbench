from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from copper_mvp.access import Principal
from copper_mvp.api import create_app
from copper_mvp.common import WorkbenchError
from copper_mvp.data import DataRepository
from copper_mvp.research_service import ResearchService
from copper_mvp.research_store import ResearchStore
from copper_mvp.storage import RunStore


@pytest.fixture(scope="module")
def data():
    return DataRepository()


def provider(request):
    payload = json.loads(request.content)
    feedback = [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]
    if not feedback or "error" in feedback[-1]:
        name, arguments = "project_status", {"purpose": "核对当前可用方法"}
    else:
        name = "finish_answer"
        arguments = {"kind": "answer", "answer": "当前有六种注册预测方法和三种优化方法。",
                     "evidence_ids": [feedback[-1]["evidence_id"]]}
    return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "reasoning_content": "synthetic-private-provider-context",
        "tool_calls": [{"id": "tool-call", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 100}})


def login(client):
    key = client.app.state.workbench.access.owner_key_path.read_text().strip()
    response = client.post("/api/auth/session", json={"access_code": key})
    assert response.status_code == 200


def session(client):
    response = client.post("/api/v2/sessions", json={"title": "合成流程验证"})
    assert response.status_code == 201
    return response.json()["id"]


def wait_task(client, identifier, wanted=None):
    for _ in range(200):
        task = client.get("/api/v2/tasks/" + identifier).json()
        if task["status"] == wanted or task["status"] in ("completed", "failed", "cancelled", "needs_attention", "paused"):
            return task
        time.sleep(0.025)
    raise AssertionError("task did not reach its boundary")


def test_authenticated_free_questions_followup_and_scoped_access(tmp_path, data):
    with TestClient(create_app(tmp_path, data), base_url="http://127.0.0.1") as client:
        assert client.get("/api/overview").status_code == 401
        assert client.post("/api/auth/session", json={"access_code": "wrong"}).status_code == 401
        login(client)
        client.app.state.workbench.research.transport = httpx.MockTransport(provider)
        identifier = session(client)
        first = client.post(f"/api/v2/sessions/{identifier}/messages", json={
            "question": "现在这套系统有哪些方法？", "request_key": "first", "max_cost_cny": 0.5}).json()
        task = wait_task(client, first["id"])
        assert task["status"] == "completed"
        assert task["model_calls"] == 2 and task["tool_calls"] == 1
        assert task["request"]["settings"]["max_cost_cny"] == 0.5
        assert task["evidence"] and task["usage"]["spent_cny"] > 0
        repeated = client.post(f"/api/v2/sessions/{identifier}/messages", json={
            "question": "现在这套系统有哪些方法？", "request_key": "first", "max_cost_cny": 0.5}).json()
        assert repeated["reused"] and repeated["id"] == first["id"]
        second = client.post(f"/api/v2/sessions/{identifier}/messages", json={
            "question": "那优化方法呢？", "request_key": "followup"}).json()
        assert wait_task(client, second["id"])["status"] == "completed"
        assert len(client.get(f"/api/v2/sessions/{identifier}").json()["messages"]) == 4
        other_key = client.post("/api/v2/access-keys", json={"user_id": "other", "role": "viewer"}).json()["access_code"]
        headers = {"Authorization": "Bearer " + other_key}
        assert client.get(f"/api/v2/sessions/{identifier}", headers=headers).status_code == 404
        assert client.get(f"/api/v2/tasks/{first['id']}", headers=headers).status_code == 404
        assert client.post("/api/runs", json={"task_type": "train", "request_key": "forbidden"}, headers=headers).status_code == 403
        with client.app.state.workbench.store.connection() as c:
            saved = "\n".join(c.iterdump())
        assert "synthetic-private-provider-context" not in saved
        assert "data:" in client.get(f"/api/v2/tasks/{first['id']}/events").text
        cookie = client.cookies.get("culab_session")
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/overview", headers={"Cookie": "culab_session=" + cookie}).status_code == 401


def test_pause_restart_resume_reuses_evidence_and_preserves_budget(tmp_path, data, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    from copper_mvp.research_tools import ResearchTools
    original = ResearchTools.project_status
    def slow(self):
        entered.set()
        assert release.wait(5)
        return original(self)
    monkeypatch.setattr(ResearchTools, "project_status", slow)
    with TestClient(create_app(tmp_path, data), base_url="http://127.0.0.1") as client:
        login(client)
        wb = client.app.state.workbench
        wb.research.transport = httpx.MockTransport(provider)
        identifier = session(client)
        task_id = client.post(f"/api/v2/sessions/{identifier}/messages", json={
            "question": "查看可用方法", "request_key": "pause"}).json()["id"]
        assert entered.wait(5)
        current = client.get("/api/v2/tasks/" + task_id).json()
        paused = client.post(f"/api/v2/tasks/{task_id}/pause", json={"expected_version": current["version"]})
        assert paused.status_code == 200
        release.set()
        current = wait_task(client, task_id, "paused")
        assert current["status"] == "paused" and len(current["evidence"]) == 1
        wb.research.close()
        wb.research = ResearchService(wb, wb.access, transport=httpx.MockTransport(provider))
        response = client.post(f"/api/v2/tasks/{task_id}/resume", json={"expected_version": current["version"]})
        assert response.status_code == 200
        result = wait_task(client, task_id)
        assert result["status"] == "completed"
        assert result["cache_hits"] == 1 and result["model_calls"] == 3
        assert result["result"]["provider_context_restarted"]
        entered.clear(); release.clear()
        cancellation = client.post(f"/api/v2/sessions/{identifier}/messages", json={
            "question": "再核对一次", "request_key": "cancel"}).json()["id"]
        assert entered.wait(5)
        current = client.get("/api/v2/tasks/" + cancellation).json()
        assert client.post(f"/api/v2/tasks/{cancellation}/cancel", json={"expected_version": current["version"]}).status_code == 200
        release.set()
        cancelled = wait_task(client, cancellation)
        assert cancelled["status"] == "cancelled" and cancelled["result"] is None


def test_expired_worker_cannot_commit_and_unknown_model_charge_is_not_retried(tmp_path, data):
    store = ResearchStore(RunStore(tmp_path / "leases"))
    actor = Principal("owner", "owner")
    conversation = store.create_session(actor, "lease", {})
    task, _ = store.create_task(actor, conversation["id"], {"question": "q", "request_key": "lease", "context": {}, "source_version": "v"})
    old = store.claim(task["id"], "first", lease_seconds=0.01)
    time.sleep(0.02)
    fresh = store.claim(task["id"], "second")
    assert fresh > old
    with pytest.raises(WorkbenchError, match="租约"):
        store.finish(task["id"], "first", old, {"answer": "stale"}, 1)
    store.finish(task["id"], "second", fresh, {"answer": "current"}, 2)
    calls = []
    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("synthetic timeout")
    with TestClient(create_app(tmp_path / "network", data), base_url="http://127.0.0.1") as client:
        login(client)
        client.app.state.workbench.research.transport = httpx.MockTransport(timeout)
        identifier = session(client)
        task_id = client.post(f"/api/v2/sessions/{identifier}/messages", json={
            "question": "检查当前模型", "request_key": "timeout"}).json()["id"]
        result = wait_task(client, task_id)
        assert result["status"] == "needs_attention"
        assert len(calls) == result["model_calls"] == 1
        assert result["usage"]["unsettled_cny"] > 0 and result["usage"]["spent_cny"] == 0


def test_local_event_values_are_rendered_without_being_sent_to_provider(tmp_path, data, monkeypatch):
    from copper_mvp.research_tools import ResearchTools
    def context(self, reference="selected"):
        return {"summary": {"local_fact_names": ["current_cu"]},
                "local_facts": {"current_cu": {"value": 731.246, "unit": "g/L"}}}
    monkeypatch.setattr(ResearchTools, "event_context", context)
    sent = []
    def model(request):
        body = json.loads(request.content)
        sent.append(request.content.decode())
        feedback = [json.loads(m["content"]) for m in body["messages"] if m["role"] == "tool"]
        if not feedback:
            name, args = "event_context", {"purpose": "读取当前化验值"}
        else:
            reference = feedback[-1]["evidence_id"]
            name, args = "finish_answer", {"answer": "当前 Cu 为 {{" + reference + ".current_cu}}。",
                                           "evidence_ids": [reference], "kind": "answer"}
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "synthetic", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})
    with TestClient(create_app(tmp_path, data), base_url="http://127.0.0.1") as client:
        login(client)
        client.app.state.workbench.research.transport = httpx.MockTransport(model)
        identifier = session(client)
        task_id = client.post(f"/api/v2/sessions/{identifier}/messages", json={
            "question": "当前 Cu 浓度是多少？", "request_key": "local-value"}).json()["id"]
        task = wait_task(client, task_id)
        assert task["status"] == "completed"
        assert "731.246 g/L" in task["result"]["answer"]
        assert all("731.246" not in payload for payload in sent)
