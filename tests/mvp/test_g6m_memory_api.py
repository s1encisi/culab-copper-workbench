from __future__ import annotations

import json
import time

import httpx
from fastapi.testclient import TestClient

from copper_mvp.api import create_app
from copper_mvp.data import DataRepository


def test_document_answer_and_memory_revalidate_source_without_provider_disclosure(tmp_path):
    sent = []
    marker = "SYNTHETIC-PROTECTED-SOP-BODY-8249"
    def provider(request):
        payload = json.loads(request.content)
        sent.append(payload)
        feedback = [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]
        if not feedback or "error" in feedback[-1]:
            name, arguments = "search_documents", {"query": "铜浓度单位", "purpose": "读取本地规程依据"}
        else:
            name = "finish_answer"
            reference = feedback[-1]["evidence_id"]
            arguments = {"kind": "answer", "answer": "本地文档依据：{{" + reference + ".document_excerpt_1}}",
                         "evidence_ids": [reference]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "synthetic-call", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})

    with TestClient(create_app(tmp_path, DataRepository()), base_url="http://127.0.0.1") as client:
        code = client.app.state.workbench.access.owner_key_path.read_text().strip()
        assert client.post("/api/auth/session", json={"access_code": code}).status_code == 200
        client.app.state.workbench.research.transport = httpx.MockTransport(provider)
        document = {"doc_id": "sop-fixture", "title": "合成规程", "author": "synthetic-author",
                    "source": "synthetic fixture", "authorization": "local test only", "license": "synthetic",
                    "version_label": "v1", "format": "md", "effective_at": "2025-01-01T00:00:00Z"}
        assert client.post("/api/v2/knowledge/documents", json=document).status_code == 201
        path = "/api/v2/knowledge/documents/sop-fixture/versions/1"
        assert client.put(path + "/content", content=(marker + "。铜浓度单位为 g/L。").encode()).status_code == 200
        assert client.post(path + "/index").status_code == 200
        conversation = client.post("/api/v2/sessions", json={"title": "文档会话"}).json()["id"]
        task = client.post(f"/api/v2/sessions/{conversation}/messages",
                           json={"question": "查阅铜浓度单位的文档依据", "request_key": "document", "max_cost_cny": 0.5})
        assert task.status_code == 202, task.text
        identifier = task.json()["id"]
        for _ in range(600):
            state = client.get("/api/v2/tasks/" + identifier).json()
            if state["status"] in ("completed", "failed", "needs_attention"):
                break
            time.sleep(0.025)
        assert state["status"] == "completed", state.get("error")
        assert marker in state["result"]["answer"]
        assert marker not in json.dumps(sent, ensure_ascii=False)
        assert len(sent) == 2
        captured = client.post(f"/api/v2/sessions/{conversation}/memory", json={"ttl_hours": 24})
        assert captured.status_code == 200, captured.text
        evidence = captured.json()["summary"]["tasks"][0]["evidence"][0]
        assert evidence["usable"]
        assert marker not in json.dumps(captured.json(), ensure_ascii=False)
        with client.app.state.workbench.store.connection() as c:
            research_database = "\n".join(c.iterdump())
        assert marker not in research_database
        forged = client.post(f"/api/v2/sessions/{conversation}/memory", json={"ttl_hours": 24, "approved": True})
        assert forged.status_code == 422
        assert client.delete("/api/v2/knowledge/documents/sop-fixture").status_code == 200
        changed = client.get("/api/v2/tasks/" + identifier).json()
        assert marker not in changed["result"]["answer"]
        assert changed["result"]["document_reference_issues"]
        history = client.get(f"/api/v2/sessions/{conversation}").json()
        assert marker not in json.dumps(history, ensure_ascii=False)
        memory = client.get(f"/api/v2/sessions/{conversation}/memory").json()
        assert memory["status"] == "references_changed"
        assert not memory["summary"]["tasks"][0]["evidence"][0]["usable"]
        assert memory["summary"]["tasks"][0]["evidence"][0]["reference_issues"]


def test_document_revocation_invalidates_repeated_tool_cache(tmp_path):
    from copper_mvp.access import Principal
    from copper_mvp.knowledge_contracts import DocumentAccess
    calls, holder = [], {}
    def provider(request):
        payload = json.loads(request.content)
        calls.append(payload)
        feedback = [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]
        if len(calls) == 2:
            holder["wb"].knowledge.change_access(Principal("owner", "owner"), "cache-fixture", DocumentAccess(revoked=True))
        if len(calls) < 3:
            name, arguments = "search_documents", {"query": "浓度单位", "purpose": "读取当前文档"}
        else:
            assert feedback[-1]["data"]["items"] == []
            name, arguments = "finish_answer", {"kind": "clarification", "answer": "文档已撤销，请重新选择可访问的来源。",
                                               "evidence_ids": []}
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "synthetic-cache-call", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})
    with TestClient(create_app(tmp_path, DataRepository()), base_url="http://127.0.0.1") as client:
        holder["wb"] = client.app.state.workbench
        key = holder["wb"].access.owner_key_path.read_text().strip()
        assert client.post("/api/auth/session", json={"access_code": key}).status_code == 200
        holder["wb"].research.transport = httpx.MockTransport(provider)
        spec = {"doc_id": "cache-fixture", "title": "合成缓存验证", "author": "fixture",
                "source": "synthetic", "authorization": "local test", "license": "synthetic",
                "version_label": "v1", "format": "md", "effective_at": "2025-01-01T00:00:00Z"}
        assert client.post("/api/v2/knowledge/documents", json=spec).status_code == 201
        base = "/api/v2/knowledge/documents/cache-fixture/versions/1"
        assert client.put(base + "/content", content="浓度单位为 g/L。".encode()).status_code == 200
        assert client.post(base + "/index").status_code == 200
        session = client.post("/api/v2/sessions", json={"title": "缓存撤销"}).json()["id"]
        task = client.post(f"/api/v2/sessions/{session}/messages",
                           json={"question": "检查当前文档", "request_key": "cache"}).json()
        for _ in range(600):
            result = client.get("/api/v2/tasks/" + task["id"]).json()
            if result["status"] in ("completed", "failed", "needs_attention"):
                break
            time.sleep(0.025)
        assert result["status"] == "completed", result.get("error")
        assert result["cache_hits"] == 0
        assert len(calls) == 3
