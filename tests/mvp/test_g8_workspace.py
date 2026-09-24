from fastapi.testclient import TestClient

from copper_mvp.api import create_app
from copper_mvp.data import DataRepository


def test_workspace_summaries_use_current_actor_and_real_records(tmp_path):
    with TestClient(create_app(tmp_path, DataRepository()), base_url="http://127.0.0.1") as client:
        wb = client.app.state.workbench
        owner_code = wb.access.owner_key_path.read_text(encoding="utf-8").strip()
        assert client.post("/api/auth/session", json={"access_code": owner_code}).status_code == 200
        owner = wb.access.authenticate(key=owner_code)
        key = client.post("/api/v2/access-keys", json={"user_id": "reader", "role": "researcher"}).json()["access_code"]
        reader = wb.access.authenticate(key=key)
        identities = []
        for actor, label in ((owner, "owner-private-topic"), (reader, "reader-topic")):
            session = wb.research.store.create_session(actor, label, {})
            task, _ = wb.research.store.create_task(
                actor,
                session["id"],
                {
                    "question": label,
                    "request_key": label,
                    "context": {},
                    "source_version": "synthetic",
                    "role": actor.role,
                    "auth_ref": actor.key_hash,
                },
            )
            fence = wb.research.store.claim(task["id"], "fixture")
            wb.research.store.finish(task["id"], "fixture", fence, {"answer": "saved", "model_answer": "saved"}, 1)
            identities.append(task["id"])
        headers = {"Authorization": "Bearer " + key}
        summary = client.get("/api/v2/workspace", headers=headers)
        assert summary.status_code == 200
        assert summary.json()["budget"]["scope"] == "own_research"
        assert summary.json()["user"] == {"id": "reader", "role": "researcher"}
        tasks = client.get("/api/v2/workspace/tasks", headers=headers).json()["items"]
        assert any(item["id"] == identities[1] for item in tasks)
        assert all(item["id"] != identities[0] for item in tasks)
        assert "owner-private-topic" not in str(tasks)
        assert (
            client.get("/api/v2/workspace/research-tasks/" + identities[0] + "/events", headers=headers).status_code
            == 404
        )
        assert (
            client.get("/api/v2/workspace/research-tasks/" + identities[1] + "/events", headers=headers).status_code
            == 200
        )
        audit = client.get("/api/v2/workspace/governance", headers=headers).json()
        assert all(row["task_id"] != identities[0] for row in audit["events"])
        owner_release = client.post(
            "/api/v2/releases",
            json={
                "request_key": "owner-proposal",
                "candidate_id": "builtin-persistence",
                "target": "cu",
                "expected_version": 0,
                "reason": "synthetic list isolation",
            },
        ).json()
        reader_release = client.post(
            "/api/v2/releases",
            headers=headers,
            json={
                "request_key": "reader-proposal",
                "candidate_id": "builtin-persistence",
                "target": "as",
                "expected_version": 0,
                "reason": "synthetic list isolation",
            },
        ).json()
        visible = client.get("/api/v2/releases", headers=headers)
        assert visible.status_code == 200
        assert [row["id"] for row in visible.json()["items"]] == [reader_release["id"]]
        assert all("proposer_auth" not in row for row in visible.json()["items"])
        owner_visible = client.get("/api/v2/releases").json()["items"]
        assert {row["id"] for row in owner_visible} == {owner_release["id"], reader_release["id"]}
        client.cookies.clear()
        assert client.get("/api/v2/workspace").status_code == 401


def test_frontend_entry_revalidates_after_a_new_build(tmp_path):
    frontend = tmp_path / "ui"
    frontend.mkdir()
    index = frontend / "index.html"
    index.write_text("<html>first-build</html>", encoding="utf-8")
    with TestClient(
        create_app(tmp_path / "runtime", DataRepository(), frontend_dir=frontend), base_url="http://127.0.0.1"
    ) as client:
        first = client.get("/models")
        assert first.status_code == 200
        assert first.headers["cache-control"] == "no-cache"
        index.write_text("<html>updated-second-build</html>", encoding="utf-8")
        second = client.get("/models", headers={"If-None-Match": first.headers["etag"]})
        assert second.status_code == 200
        assert "updated-second-build" in second.text
        assert second.headers["etag"] != first.headers["etag"]


def test_stopped_calibration_is_reported_without_rewriting_saved_evidence(tmp_path, monkeypatch):
    import json

    import copper_mvp.model_comparisons as jobs

    monkeypatch.setattr(jobs, "process_alive", lambda pid: False)
    with TestClient(create_app(tmp_path, DataRepository()), base_url="http://127.0.0.1") as client:
        wb = client.app.state.workbench
        client.post(
            "/api/auth/session", json={"access_code": wb.access.owner_key_path.read_text(encoding="utf-8").strip()}
        )
        identifier = "d" * 32
        folder = wb.root / "calibration_studies" / identifier
        folder.mkdir(parents=True)
        payload = {
            "id": identifier,
            "owner_pid": 123456,
            "status": "running",
            "created_at": "2026-01-01T00:00:00+00:00",
            "request": {},
        }
        saved = json.dumps(payload).encode()
        (folder / "state.json").write_bytes(saved)
        assert client.get("/api/v2/calibrations/" + identifier).json()["status"] == "interrupted"
        assert client.get("/api/v2/calibrations").json()["items"][0]["status"] == "interrupted"
        tasks = client.get("/api/v2/workspace/tasks").json()["items"]
        assert next(row for row in tasks if row["id"] == identifier)["status"] == "interrupted"
        assert (folder / "state.json").read_bytes() == saved
