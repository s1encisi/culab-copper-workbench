from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from copper_mvp.api import create_app
from copper_mvp.checks import verify_prediction
from copper_mvp.common import WorkbenchError
from copper_mvp.contracts import RunRequest
from copper_mvp.data import DataRepository
from copper_mvp.modeling import ModelManager
from copper_mvp.optimization import solve
from copper_mvp.storage import RunStore
from copper_mvp.workflows import Workbench


@pytest.fixture(scope="session")
def data():
    return DataRepository()


@pytest.fixture(scope="session")
def trained(tmp_path_factory, data):
    root = tmp_path_factory.mktemp("trained")
    manager = ModelManager(root, data)
    manifest = manager.train("a" * 32, lambda *args: None)
    return manager, manifest


@pytest.fixture
def client(tmp_path, data):
    with TestClient(create_app(tmp_path, data, enforce_auth=False)) as c:
        yield c


def wait_run(client, run_id):
    for _ in range(400):
        value = client.get("/api/runs/" + run_id).json()
        if value["status"] in ("completed", "failed"):
            return value
        time.sleep(0.025)
    raise AssertionError("run timed out")


def test_actual_data_and_candidate_feature_consistency(data):
    X, y = data.training_data()
    assert X.shape == (2953, 114) and y.shape == (2953, 2)
    assert len(data.fold_for) == 2732
    event = data.events(scope="dual", limit=1)["items"][0]
    row = data.row(event["event_id"])
    currents = np.array([[row.stage3_current_a__t_minus_0h, row.stage4_current_a__t_minus_0h]])
    candidate = data.candidate_matrix(event["event_id"], [3, 4], currents)
    np.testing.assert_allclose(
        candidate, data.X.loc[[event["event_id"]]].to_numpy(), rtol=1e-8, atol=1e-5, equal_nan=True
    )
    changed = data.candidate_matrix(event["event_id"], [3, 4], currents * 1.01)
    for j, stage in enumerate((3, 4)):
        tag = f"stage{stage}_current_a"
        mean_index = data.feature_columns.index(tag + "__mean_12h")
        count = row[tag + "__valid_count_12h"]
        assert changed[0, mean_index] - candidate[0, mean_index] == pytest.approx(currents[0, j] * 0.01 / count)
    assert not any(c.startswith(("target_", "next_")) for c in data.feature_columns)


def test_full_training_and_oof_model_scope(data, trained):
    manager, manifest = trained
    assert len(manifest["artifacts"]) == 24
    assert len(manifest["metrics"]) == 6 and manifest["oof_events"] == 2732
    for metric in manifest["metrics"]:
        assert np.isfinite(metric["mae"]) and metric["n"] == 2732
    event = data.events(scope="dual", limit=1)["items"][0]
    prediction = manager.predict(event["event_id"])
    assert prediction["fold_id"] == data.fold_for[event["event_id"]]
    assert verify_prediction(prediction, data.context(event["event_id"]))["passed"]
    assert prediction["fit_cutoff_at"] <= prediction["decision_at"]


def test_real_optimization_recomputes_power_and_as(data, trained, tmp_path):
    manager, _ = trained
    event = data.events(scope="dual", limit=1)["items"][0]
    result = solve(
        data,
        manager,
        RunRequest(task_type="optimize", request_key="test-opt", event_id=event["event_id"], evaluation_budget=128),
        tmp_path,
        lambda *args: None,
    )
    assert result["search_evaluations"] <= 128
    row = data.row(event["event_id"])
    for c in result["candidates"]:
        power = (
            sum(
                row[f"stage{s}_voltage_v__t_minus_0h"] * c["variables"][f"stage{s}_current_a"]
                for s in result["supported_stages"]
            )
            / 1000
        )
        assert c["f2"] == pytest.approx(power)
        assert c["as"] <= result["reference"]["as"] + 1e-4
        assert c["audit_passed"]
    assert result["mode"] == "plant" and result["fold_id"] == event["fold_id"]


def test_tampered_model_rejected_after_cache_load(data, trained):
    manager, manifest = trained
    event = data.events(scope="dual", limit=1)["items"][0]
    manager.resolve(event["event_id"], "DeltaHGB", "oof_replay")
    artifact = manifest["artifacts"][f"{event['fold_id']}:DeltaHGB:cu"]
    path = manager.root / manifest["bundle_id"] / artifact["file"]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tamper")
        with pytest.raises(WorkbenchError, match="校验失败"):
            manager.resolve(event["event_id"], "DeltaHGB", "oof_replay")
    finally:
        path.write_bytes(original)


def test_api_rejects_future_fields_and_remote_origin(client, data):
    event_id = data.summaries[-1]["event_id"]
    assert (
        client.post(
            "/api/runs",
            json={"task_type": "predict", "request_key": "future", "event_id": event_id, "target_cu_g_l": 1},
        ).status_code
        == 422
    )
    assert client.get("/api/overview", headers={"Origin": "https://example.com"}).status_code == 403
    response = client.get("/api/events/" + event_id).json()
    assert response["event_id"] == event_id
    assert "target_cu_g_l" not in json.dumps(response)


def test_api_version_matches_project_metadata(client):
    import tomllib

    from copper_mvp.common import PROJECT_ROOT

    expected = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert client.get("/api/health").json()["version"] == expected
    assert client.get("/openapi.json").json()["info"]["version"] == expected


def test_api_benchmark_selection_explanation_and_export(client):
    payload = {"task_type": "optimize", "mode": "benchmark", "request_key": "bench", "evaluation_budget": 130}
    initial = client.post("/api/runs", json=payload)
    assert initial.status_code == 202
    run = wait_run(client, initial.json()["run_id"])
    assert run["status"] == "completed", run["error"]
    result = run["result"]
    assert result["search_evaluations"] <= 130 and result["candidates"]
    for c in result["candidates"]:
        x, y = c["variables"]["u1"], c["variables"]["u2"]
        assert c["f1"] == pytest.approx(x * x + y * y)
        assert c["f2"] == pytest.approx((x - 2) ** 2 + (y - 2) ** 2)
        assert x + y <= 3 + 1e-7 and c["as"] is None
    candidate = result["candidates"][0]
    selected = client.post(
        f"/api/runs/{run['run_id']}/selection", json={"candidate_id": candidate["id"], "label": "测试候选"}
    )
    assert selected.json()["selections"][0]["label"] == "测试候选"
    explained = client.post(f"/api/runs/{run['run_id']}/explanation", json={"question": "summary"}).json()
    assert explained["mode"] == "local" and "数学测试" in explained["text"]
    assert client.get(f"/api/runs/{run['run_id']}/export?format=csv").status_code == 200
    replay = client.post("/api/runs", json=payload).json()
    assert replay["run_id"] == run["run_id"] and replay["reused"]
    assert len(replay["trace"]) == len(run["trace"])
    assert client.post("/api/runs", json={**payload, "seed": 7}).status_code == 409


@pytest.mark.parametrize(
    "scenario,blocked",
    [
        ("clean", False),
        ("future_field", True),
        ("model_mismatch", True),
        ("invalid_numeric", True),
        ("invalid_interval", True),
    ],
)
def test_actual_diagnostic_paths(client, scenario, blocked):
    created = client.post(
        "/api/runs", json={"task_type": "diagnostic", "scenario": scenario, "request_key": scenario}
    ).json()
    result = wait_run(client, created["run_id"])
    assert result["status"] == "completed", result["error"]
    assert result["result"]["blocked"] is blocked and result["result"]["passed"]


def test_concurrent_idempotency_and_budget(tmp_path):
    store = RunStore(tmp_path)
    payload = {"request_key": "one", "task_type": "predict"}
    with ThreadPoolExecutor(max_workers=6) as pool:
        created = list(pool.map(lambda _: store.create(payload, "fingerprint"), range(12)))
    assert len({r[0]["run_id"] for r in created}) == 1
    assert sum(not r[1] for r in created) == 1
    store.reserve_cost("a", "2026-09", 0.7, 1)
    with pytest.raises(WorkbenchError):
        store.reserve_cost("b", "2026-09", 0.4, 1)
    store.settle_cost("a", 0.1)
    store.reserve_cost("b", "2026-09", 0.4, 1)
    with pytest.raises(WorkbenchError):
        store.reserve_cost("c", "2026-09", float("nan"), 1)


def test_completed_runs_survive_restart(tmp_path, data):
    wb = Workbench(tmp_path, data)
    request = RunRequest(
        task_type="predict", request_key="persist", event_id=data.summaries[-1]["event_id"], model_profile="Persistence"
    )
    created = wb.submit(request)
    wb.futures[created["run_id"]].result(timeout=30)
    original = wb.store.get(created["run_id"])
    interrupted, _ = wb.store.create({"task_type": "predict", "request_key": "interrupted"}, "x")
    wb.close()
    restored = Workbench(tmp_path, data)
    try:
        value = restored.store.get(created["run_id"])
        assert value["result"] == original["result"] and value["status"] == "completed"
        assert restored.submit(request)["reused"]
        assert restored.store.get(interrupted["run_id"])["error"]["code"] == "INTERRUPTED"
    finally:
        restored.close()


def test_llm_metadata_only_and_cached_budget(client, monkeypatch):
    # All credentials here are synthetic; HTTP traffic uses MockTransport.
    for key, value in {
        "COPPER_MVP_LIVE_LLM": "1",
        "COPPER_MVP_MONTHLY_CNY": "2",
        "COPPER_MVP_CALL_CAP_CNY": "1",
        "COPPER_MVP_INPUT_CNY_PER_MILLION": "2",
        "COPPER_MVP_OUTPUT_CNY_PER_MILLION": "4",
        "DEEPSEEK_MODEL": "deepseek-v4-pro",
        "DEEPSEEK_API_KEY": "sk-test-synthetic-not-real",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    }.items():
        monkeypatch.setenv(key, value)
    created = client.post(
        "/api/runs", json={"task_type": "diagnostic", "request_key": "explain", "scenario": "clean"}
    ).json()
    run = wait_run(client, created["run_id"])
    calls = []

    def response(request):
        body = request.content.decode()
        assert "origin_cu_g_l" not in body and "origin_as_mg_l" not in body
        calls.append(body)
        value = {
            "artifact_ids": [run["run_id"]],
            "summary": "本次任务已经完成，计算依据可以从运行记录追溯。",
            "reason_codes": ["RUN_COMPLETED"],
            "changed_numeric_result": False,
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(value, ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    service = client.app.state.workbench.explanations
    a = service.explain(run["run_id"], "summary", True, transport=httpx.MockTransport(response))
    b = service.explain(run["run_id"], "summary", True, transport=httpx.MockTransport(response))
    assert a["mode"] == "llm" and b["cached"] and len(calls) == 1
