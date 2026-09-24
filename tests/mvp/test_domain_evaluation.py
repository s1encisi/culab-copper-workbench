import json
import time

import httpx
import numpy as np
from fastapi.testclient import TestClient

from copper_mvp.api import create_app
from copper_mvp.data import DataRepository
from copper_mvp.research_tools import ResearchTools


class TestEmbedder:
    signature = "synthetic-domain-embedding-v1"

    def encode(self, texts, query=False):
        return np.ones((len(texts), 4), dtype=np.float32)


def login(client):
    wb = client.app.state.workbench
    client.post("/api/auth/session", json={"access_code": wb.access.owner_key_path.read_text(encoding="utf-8").strip()})
    wb.knowledge.embedder = TestEmbedder()
    return wb


def responder(sent):
    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        docs = any(v["function"]["name"] == "read_document" for v in payload["tools"])
        feedback = [json.loads(v["content"]) for v in payload["messages"] if v["role"] == "tool"]
        if not feedback:
            name, args = (
                ("read_document", {"purpose": "读取评测资料"}) if docs else ("project_status", {"purpose": "核对目录"})
            )
        else:
            assert "error" not in feedback[-1], feedback[-1]
            identifier = feedback[-1]["evidence_id"]
            name, args = (
                "finish_answer",
                {"kind": "answer", "answer": "测试回答：已根据证据完成核对。", "evidence_ids": [identifier]},
            )
            if docs:
                args["document_refs"] = [{"evidence_id": identifier, "field": "document_excerpt"}]
        return httpx.Response(
            200,
            json={
                "model": "synthetic-provider",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "fixture",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 0},
            },
        )

    return respond


def test_domain_comparison_reuses_research_records_and_scoped_budget(tmp_path):
    with TestClient(create_app(tmp_path, DataRepository()), base_url="http://127.0.0.1") as client:
        wb = login(client)
        sent = []
        wb.research.transport = httpx.MockTransport(responder(sent))
        request = {"request_key": "domain-fixture"}
        created = client.post("/api/v2/domain-evaluations", json=request)
        assert created.status_code == 201, created.text
        study = created.json()
        identifier = study["id"]
        assert study["status"] == "planned" and not sent
        assert client.post("/api/v2/domain-evaluations", json=request).json()["id"] == identifier
        assert client.post("/api/v2/domain-evaluations/" + identifier + "/run", json={}).status_code == 202
        for _ in range(400):
            current = client.get("/api/v2/domain-evaluations/" + identifier).json()
            if current["status"] not in ("queued", "running"):
                break
            time.sleep(0.025)
        assert current["status"] == "completed", current
        assert len(current["cases"]) == 4
        assert current["provider_mode"] == "test_transport"
        assert all(v["status"] == "completed" for v in current["cases"])
        assert all(v["max_tokens"] == 2048 and v["thinking"]["type"] == "disabled" for v in sent)
        actor = wb.access.authenticate(key=wb.access.owner_key_path.read_text(encoding="utf-8").strip())
        compact = ResearchTools(wb, actor, {}).project_status()
        assert compact["targets"] == [{"name": "cu", "unit": "g/L"}, {"name": "as", "unit": "mg/L"}]
        assert len(json.dumps(compact)) < 7000
        assert all(set(row) == {"method_id", "status"} for row in compact["models"]["items"])
        summary = {v["variant"]: v for v in current["summary"]}
        assert summary["prompt"]["document_citations"] == 0
        assert summary["rag"]["document_citations"] == 2
        assert sum(v["cost_cny"] + v["unsettled_cny"] for v in current["summary"]) <= 2
        calls = len(sent)
        client.post("/api/v2/domain-evaluations/" + identifier + "/run", json={})
        assert len(sent) == calls
        reviewed = client.post(
            "/api/v2/domain-evaluations/" + identifier + "/review",
            json={"verdict": "accepted", "notes": "测试替身流程核对完成"},
        )
        assert reviewed.status_code == 200 and reviewed.json()["review"]["verdict"] == "accepted"
        saved = json.loads((tmp_path / "domain_evaluations" / identifier / "state.json").read_text(encoding="utf-8"))
        assert all("result" not in row for row in saved["cases"])
        reader = wb.access.issue(
            wb.access.authenticate(key=wb.access.owner_key_path.read_text(encoding="utf-8").strip()),
            "reader",
            "researcher",
        )
        assert (
            client.get(
                "/api/v2/domain-evaluations/" + identifier, headers={"Authorization": "Bearer " + reader}
            ).status_code
            == 404
        )


def test_domain_plan_is_local_until_model_execution_is_enabled(tmp_path):
    with TestClient(create_app(tmp_path, DataRepository()), base_url="http://127.0.0.1") as client:
        wb = login(client)
        wb.research.allow_live = False
        created = client.post("/api/v2/domain-evaluations", json={"request_key": "local-plan"})
        assert created.status_code == 201, created.text
        identifier = created.json()["id"]
        assert not client.get("/api/v2/domain-evaluations/defaults").json()["can_run"]
        response = client.post("/api/v2/domain-evaluations/" + identifier + "/run", json={})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "DOMAIN_PROVIDER_DISABLED"
        assert (
            client.post(
                "/api/v2/domain-evaluations", json={"request_key": "local-plan", "title": "changed"}
            ).status_code
            == 409
        )


def test_failed_case_does_not_abort_or_repeat_the_remaining_comparison(tmp_path):
    with TestClient(create_app(tmp_path, DataRepository()), base_url="http://127.0.0.1") as client:
        wb = login(client)
        sent = []
        answer = responder(sent)
        calls = []

        def provider(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(400, json={"error": {"message": "synthetic rejected request"}})
            return answer(request)

        wb.research.transport = httpx.MockTransport(provider)
        study = client.post(
            "/api/v2/domain-evaluations", json={"request_key": "failed-baseline", "questions": ["Cu 的单位？"]}
        ).json()
        url = "/api/v2/domain-evaluations/" + study["id"]
        assert client.post(url + "/run", json={}).status_code == 202
        for _ in range(400):
            result = client.get(url).json()
            if result["status"] not in ("queued", "running"):
                break
            time.sleep(0.025)
        assert result["status"] == "completed", result
        assert [row["status"] for row in result["cases"]] == ["failed", "completed"]
        before = len(calls)
        client.post(url + "/run", json={})
        assert len(calls) == before
