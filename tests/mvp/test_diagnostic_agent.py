from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from copper_mvp.api import create_app
from copper_mvp.common import WorkbenchError, digest, dumps
from copper_mvp.contracts import RunRequest
from copper_mvp.diagnostic_agent import AgentSettings, DiagnosisReport, DiagnosticAgentService, estimate_usage
from copper_mvp.diagnostic_tools import DiagnosticTools
from copper_mvp.storage import RunStore


class SmallData:
    """Hand-checkable local process fixture; never used as real plant evidence."""
    dataset_version = "test-data"
    def row(self, _):
        return {"stage3_voltage_v__t_minus_0h": 10, "stage4_voltage_v__t_minus_0h": 20}

    def candidate_matrix(self, event, stages, currents):
        return currents.copy()

    def training_data(self):
        return None, pd.DataFrame([[1, 10], [3, 30], [5, 50], [7, 70]])

    def training_ids(self, fold):
        return [0, 1, 2, 3]


class SmallModels:
    def resolve(self, *args):
        return {"names": {"cu": "DeltaHGB", "as": "DeltaHGB"}, "fold_id": "FOLD_1"}

    def predict_matrix(self, resolved, matrix):
        return np.column_stack((np.full(len(matrix), 5.0), matrix[:, 1]))


def make_source(store):
    request = RunRequest(task_type="optimize", request_key="source", event_id="local-event", bundle_id="a" * 32).model_dump()
    source, _ = store.create(request, "source-fingerprint")
    result = {"kind": "optimization", "mode": "plant", "event_id": "local-event", "bundle_id": "a" * 32, "fold_id": "FOLD_1", "model_scope": "oof_replay", "models": {"cu": "DeltaHGB", "as": "DeltaHGB"}, "supported_stages": [3, 4], "reference": {"f1": 5.0, "f2": 3.0, "as": 100.0, "variables": {"stage3_current_a": 100.0, "stage4_current_a": 100.0}}, "ranges": [{"stage": s, "lower": 90.0, "upper": 110.0, "varies": True, "historical_n": 100} for s in (3, 4)], "total_front_points": 1, "search_evaluations": 64, "feasible_evaluations": 32, "feasible_rate": .5, "stop_reason": "evaluation_budget", "audit": {"passed": True}}
    store.finish(source["run_id"], result, 1)
    return store.get(source["run_id"])


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path)


@pytest.fixture
def source(store):
    return make_source(store)


def test_probe_constraint_counts_and_plateau_have_hand_calculated_answer(source):
    tools = DiagnosticTools(source, SmallData(), SmallModels())
    constraints = tools.probe_constraints(9)
    assert constraints["probe_points"] == 81
    assert constraints["feasible_points"] == 45
    assert constraints["rejected_only_by_as"] == 36
    assert constraints["probe_front_points"] == 1
    response = tools.probe_response(9)
    assert response["cu_constant_in_probe"] is True
    assert response["as_constant_in_probe"] is False
    assert response["all_points"]["as_span_mg_l"] == 20
    assert response["power_span_kw"] == pytest.approx(.6)
    assert response["objective_alignment_in_feasible_probe"]["minimum_power_point_also_minimizes_cu"] is True
    assert tools.inspect_run()["original_rejection_breakdown_available"] is False


def test_single_axis_probe_does_not_change_other_current(source):
    tools = DiagnosticTools(source, SmallData(), SmallModels())
    response = tools.probe_response(9, "stage3")
    assert response["as_constant_in_probe"] is True
    assert response["power_span_kw"] == pytest.approx(.2)


def test_response_reports_tradeoff_when_quality_and_power_oppose(source):
    class OpposedModels(SmallModels):
        def predict_matrix(self, resolved, matrix):
            return np.column_stack((5 - (matrix[:, 0] - 100)*.01, matrix[:, 1]))
    tools = DiagnosticTools(source, SmallData(), OpposedModels())
    result = tools.probe_response(9, "stage3")
    assert result["probe_front_points"] == 9
    assert result["objective_alignment_in_feasible_probe"]["minimum_power_point_also_minimizes_cu"] is False


def test_fixed_range_and_reference_mismatch(source):
    for item in source["result"]["ranges"]:
        item.update(lower=100.0, upper=100.0, varies=False)
    tools = DiagnosticTools(source, SmallData(), SmallModels())
    assert tools.inspect_run()["variable_count"] == 0
    assert tools.probe_constraints()["probe_points"] == 1
    with pytest.raises(WorkbenchError, match="固定"):
        tools.probe_response(axis="stage3")
    source["result"]["reference"]["f1"] = 6.0
    with pytest.raises(WorkbenchError, match="复算"):
        DiagnosticTools(source, SmallData(), SmallModels())


def call(name, arguments):
    return {"id": "call_" + name, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def report(evidence="E2"):
    return {"summary": "有限探测中Cu模型响应保持不变，功率随电流变化；尚不能据此推断现场因果作用。", "findings": [{"claim": "本次探测出现Cu模型响应平台。", "status": "supported", "evidence_ids": [evidence]}], "next_checks": ["refine_original_grid"]}


class ScriptedProvider:
    def __init__(self, script):
        self.script = script
        self.payloads = []
        self.timeouts = []

    def __call__(self, request):
        payload = json.loads(request.content)
        self.payloads.append(payload)
        self.timeouts.append(request.extensions.get("timeout", {}))
        calls = self.script[len(self.payloads) - 1]
        return httpx.Response(200, json={"model": "deepseek-v4-flash", "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "reasoning_content": "provider-only-test-reasoning", "tool_calls": calls}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 400}})


def run_agent(store, source, monkeypatch, script, settings=None):
    monkeypatch.setattr("copper_mvp.diagnostic_agent.read_deepseek_key", lambda: SecretStr("test-secret-never-log"))
    provider = ScriptedProvider(script)
    agent = DiagnosticAgentService(store, SmallData(), SmallModels(), settings, httpx.MockTransport(provider))
    request = {"request_key": "diagnose", "task_type": "agent_diagnostic", "source_run_id": source["run_id"], "question": "why_few_candidates"}
    job, _ = store.create(request, digest(request))
    agent.execute(job["run_id"])
    return store.get(job["run_id"]), provider


def basic_script():
    return [[call("inspect_run", {"purpose": "确认原搜索范围和可行率"})], [call("probe_response", {"purpose": "检验模型是否出现平台", "points_per_axis": 9, "axis": "joint"})], [call("finish_diagnosis", report())]]


def test_real_loop_returns_tool_feedback_preserves_protocol_and_excludes_secrets(store, source, monkeypatch):
    result, provider = run_agent(store, source, monkeypatch, basic_script())
    assert result["status"] == "completed"
    assert result["result"]["usage"]["api_calls"] == 3
    assert result["result"]["usage"]["tool_calls"] == 2
    assert provider.payloads[0]["reasoning_effort"] == "high"
    assert provider.payloads[0]["max_tokens"] == 16384
    assert provider.timeouts[0]["read"] == 120
    configuration = result["result"]["configuration"]
    assert configuration["reasoning_effort"] == "high"
    assert configuration["max_output_tokens"] == 16384
    assert configuration["max_wall_seconds"] == 300
    assert configuration["max_cost_cny"] == 2.0
    assert configuration["monthly_limit_cny"] == 100.0
    assert provider.payloads[1]["messages"][-1]["role"] == "tool"
    assert provider.payloads[1]["messages"][-2]["reasoning_content"] == "provider-only-test-reasoning"
    persisted = dumps(result)
    assert "test-secret-never-log" not in persisted
    assert "provider-only-test-reasoning" not in persisted
    outbound = json.dumps(provider.payloads)
    for forbidden in ("local-event", "stage3_current_a", "stage4_current_a", source["run_id"], "DEEPSEEK_API_KEY", "target_cu_g_l"):
        assert forbidden not in outbound
    assert store.get(source["run_id"])["result"] == source["result"]


def test_bad_evidence_reference_is_returned_to_model_for_correction(store, source, monkeypatch):
    script = basic_script()
    script.insert(2, [call("finish_diagnosis", report("E999"))])
    result, provider = run_agent(store, source, monkeypatch, script)
    assert result["status"] == "completed"
    assert result["result"]["usage"]["api_calls"] == 4
    assert "REPORT_UNKNOWN_EVIDENCE" in provider.payloads[3]["messages"][-1]["content"]


def test_followup_checks_stay_in_diagnostic_scope():
    invalid = report()
    invalid["next_checks"] = ["expand_current_range"]
    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(invalid)


def test_tool_parameter_feedback_identifies_field_without_echoing_input(store, source, monkeypatch):
    script = basic_script()
    script.insert(1, [call("probe_response", {"purpose": "检查响应", "axis": "bad-untrusted-input", "points_per_axis": 9})])
    result, provider = run_agent(store, source, monkeypatch, script)
    assert result["status"] == "completed"
    feedback = json.loads(provider.payloads[2]["messages"][-1]["content"])
    assert feedback["error"]["fields"][0]["field"] == "axis"
    assert "bad-untrusted-input" not in dumps(feedback)
    assert "原电流范围" in result["result"]["report"]["next_checks"][0]


def test_duplicate_probe_reuses_numerical_evidence(store, source, monkeypatch):
    script = basic_script()
    script.insert(2, script[1])
    result, _ = run_agent(store, source, monkeypatch, script)
    assert result["status"] == "completed"
    assert result["result"]["usage"]["tool_cache_hits"] == 1
    assert len(result["result"]["evidence"]) == 2


def test_model_can_choose_constraint_probe_before_response(store, source, monkeypatch):
    script = basic_script()
    script.insert(1, [call("probe_constraints", {"purpose": "先检查As约束排除情况", "points_per_axis": 9})])
    script[-1] = [call("finish_diagnosis", report("E3"))]
    result, _ = run_agent(store, source, monkeypatch, script)
    assert result["status"] == "completed"
    assert [e["tool"] for e in result["result"]["evidence"]] == ["inspect_run", "probe_constraints", "probe_response"]


def test_limit_does_not_masquerade_as_success(store, source, monkeypatch):
    settings = AgentSettings.load().model_copy(update={"max_calls": 2})
    result, provider = run_agent(store, source, monkeypatch, basic_script()[:2], settings)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "AGENT_CALL_LIMIT"
    assert result["result"]["report"] is None
    assert len(result["result"]["evidence"]) == 2
    assert len(provider.payloads) == 2


def test_http_failure_keeps_unknown_charge_and_does_not_retry(store, source, monkeypatch):
    monkeypatch.setattr("copper_mvp.diagnostic_agent.read_deepseek_key", lambda: SecretStr("test-secret-never-log"))
    attempts = []
    def reject(request):
        attempts.append(1)
        return httpx.Response(401, json={"error": "test-secret-never-log"})
    agent = DiagnosticAgentService(store, SmallData(), SmallModels(), transport=httpx.MockTransport(reject))
    job, _ = store.create({"request_key": "failure", "task_type": "agent_diagnostic", "source_run_id": source["run_id"], "question": "why_few_candidates"}, "failure")
    agent.execute(job["run_id"])
    result = store.get(job["run_id"])
    assert result["status"] == "failed" and len(attempts) == 1
    assert result["result"]["usage"]["unsettled_reservation_cny"] > 0
    assert "test-secret-never-log" not in dumps(result)


def test_price_estimate_cache_split_and_time_period():
    pricing = AgentSettings.load().pricing
    usage = {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_cache_hit_tokens": 400}
    peak = estimate_usage(usage, pricing, datetime(2026, 9, 7, 2, tzinfo=timezone.utc))
    offpeak = estimate_usage(usage, pricing, datetime(2026, 9, 6, 2, tzinfo=timezone.utc))
    assert peak["estimated_cost_cny"] == pytest.approx((400*.1 + 600*3 + 200*9)/1e6)
    assert offpeak["estimated_cost_cny"] == pytest.approx(peak["estimated_cost_cny"]*.5)
    with pytest.raises(WorkbenchError):
        estimate_usage({}, pricing, datetime.now(timezone.utc))


@pytest.mark.parametrize("limit", ["run", "month"])
def test_budget_rejection_happens_before_network(store, source, monkeypatch, limit):
    setting = "max_cost_cny" if limit == "run" else "monthly_limit_cny"
    settings = AgentSettings.load().model_copy(update={setting: .000001})
    result, provider = run_agent(store, source, monkeypatch, [], settings)
    assert result["status"] == "failed"
    assert len(provider.payloads) == 0
    assert result["result"]["usage"]["api_calls"] == 0
    assert result["result"]["usage"]["unsettled_reservation_cny"] == 0


def test_dotenv_key_is_selective_and_does_not_mutate_environment(tmp_path, monkeypatch):
    from copper_mvp.diagnostic_agent import read_deepseek_key
    monkeypatch.setattr("copper_mvp.diagnostic_agent.PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr("copper_mvp.diagnostic_agent.WORKSPACE_ROOT", tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("UNRELATED_TEST_TOKEN", "unchanged")
    (tmp_path / ".env").write_text('UNRELATED_TEST_TOKEN=replace\nDEEPSEEK_API_KEY="test-local-key"\n', encoding="utf-8")
    key = read_deepseek_key()
    assert key.get_secret_value() == "test-local-key"
    import os
    assert os.environ["UNRELATED_TEST_TOKEN"] == "unchanged"
    assert "DEEPSEEK_API_KEY" not in os.environ
    assert "test-local-key" not in repr(key)


def test_api_submit_reuse_history_export_and_future_field_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr("copper_mvp.diagnostic_agent.read_deepseek_key", lambda: SecretStr("test-secret-never-log"))
    provider = ScriptedProvider(basic_script() + basic_script())
    with TestClient(create_app(tmp_path, SmallData())) as client:
        wb = client.app.state.workbench
        source = make_source(wb.store)
        wb.diagnostic_agent.models = SmallModels()
        wb.diagnostic_agent.transport = httpx.MockTransport(provider)
        endpoint = f"/api/runs/{source['run_id']}/diagnoses"
        response = client.post(endpoint, json={})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        wb.futures[run_id].result(timeout=20)
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "completed"
        assert client.post(endpoint, json={}).json()["reused"] is True
        assert len(provider.payloads) == 3
        assert client.get(endpoint).json()["items"][0]["run_id"] == run_id
        assert "E2" in client.get(f"/api/runs/{run_id}/export?format=md").text
        assert client.post(endpoint, json={"target_cu_g_l": 1}).status_code == 422
        assert client.post(endpoint, json={}, headers={"origin": "https://example.com"}).status_code == 403
        original_settings = wb.diagnostic_agent.settings
        wb.diagnostic_agent.settings = original_settings.model_copy(update={"reasoning_effort": "low"})
        changed = client.post(endpoint, json={}).json()
        assert changed["run_id"] != run_id and changed["reused"] is False
        wb.futures[changed["run_id"]].result(timeout=20)
        changed_result = client.get(f"/api/runs/{changed['run_id']}").json()
        assert changed_result["result"]["configuration"]["reasoning_effort"] == "low"
        wb.diagnostic_agent.settings = original_settings
        assert client.post(endpoint, json={}).json()["run_id"] == run_id
        assert len(provider.payloads) == 6
