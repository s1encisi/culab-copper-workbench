import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from copper_mas.config.settings import (
    ConfigurationError,
    load_local_env,
    load_provider_registry,
    redact_text,
    resolve_budget_runtime,
    resolve_provider_runtime,
)
from copper_mas.llm.adapters import DeepSeekV4ProAdapter, KimiK3Adapter, StructuredTask
from copper_mas.llm.client import (
    CallBudget,
    ExternalLLMBudgetExceeded,
    ExternalLLMCallsDisabled,
    call_structured,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs/llm/providers.yaml"


def test_default_runtime_needs_no_key_and_stays_offline():
    registry = load_provider_registry(CONFIG)
    runtime = resolve_provider_runtime(registry, "deepseek_executor", environ={})
    assert runtime.live_calls is False
    assert runtime.api_key is None


def test_live_runtime_requires_key():
    registry = load_provider_registry(CONFIG)
    with pytest.raises(ConfigurationError):
        resolve_provider_runtime(
            registry,
            "deepseek_executor",
            environ={"COPPER_MAS_LIVE_CALLS": "true"},
        )


def test_kimi_live_runtime_uses_confirmed_china_region_default():
    registry = load_provider_registry(CONFIG)
    runtime = resolve_provider_runtime(
        registry,
        "kimi_orchestrator",
        environ={"COPPER_MAS_LIVE_CALLS": "true", "KIMI_API_KEY": "local-test-value"},
    )
    assert runtime.base_url == "https://api.moonshot.cn/v1"
    assert runtime.model == "kimi-k3"


def test_base_url_allowlist():
    registry = load_provider_registry(CONFIG)
    with pytest.raises(ConfigurationError):
        resolve_provider_runtime(
            registry,
            "deepseek_executor",
            environ={"DEEPSEEK_BASE_URL": "https://example.invalid"},
        )


def test_redaction_never_echoes_secret_like_value():
    raw = "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz123456"
    assert "abcdefghijklmnopqrstuvwxyz" not in redact_text(raw)


def test_provider_payload_shapes_and_no_sampling_controls():
    registry = load_provider_registry(CONFIG)
    task = StructuredTask(
        task_name="mode_card",
        system_prompt="只返回合同允许的 JSON。",
        user_payload={
            "artifact_ids": ["mode-card-origin-1"],
            "schema_version": "2.0",
            "pass_fail_status": "PASSED",
            "reason_codes": [],
            "bounded_aggregate_metadata": {"warning_count": 0},
            "instruction": "整理已经计算出的原因码。",
            "requested_output": "mode explanation",
        },
        output_schema={
            "type": "object",
            "properties": {"mode": {"type": "string"}},
            "required": ["mode"],
            "additionalProperties": False,
        },
    )
    kimi = resolve_provider_runtime(
        registry,
        "kimi_orchestrator",
        environ={"KIMI_BASE_URL": "https://api.moonshot.cn/v1"},
    )
    deepseek = resolve_provider_runtime(registry, "deepseek_executor", environ={})
    kimi_payload = KimiK3Adapter().build_payload(kimi, task)
    deepseek_payload = DeepSeekV4ProAdapter().build_payload(deepseek, task)
    assert kimi_payload["response_format"]["type"] == "json_schema"
    assert kimi_payload["response_format"]["json_schema"]["strict"] is True
    assert deepseek_payload["thinking"] == {"type": "enabled"}
    for payload in (kimi_payload, deepseek_payload):
        assert "temperature" not in payload
        assert "top_p" not in payload


def test_adapter_rejects_future_fields_before_network_layer():
    with pytest.raises(ValueError):
        StructuredTask(
            task_name="bad_task",
            system_prompt="test",
            user_payload={"target_cu_g_l": 1.0},
            output_schema={"type": "object"},
        )


def test_metadata_envelope_rejects_row_level_observations():
    with pytest.raises(ValueError):
        StructuredTask(
            task_name="raw_row",
            system_prompt="test",
            user_payload={
                "artifact_ids": ["a1"],
                "schema_version": "2.0",
                "pass_fail_status": "PASSED",
                "instruction": "test",
                "requested_output": "test",
                "observations": [{"stage3_current_a": 1.0}],
            },
            output_schema={"type": "object"},
        )


class _TinyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


def _safe_task() -> StructuredTask:
    return StructuredTask(
        task_name="tiny_output",
        system_prompt="只输出JSON。",
        user_payload={
            "artifact_ids": ["artifact-1"],
            "schema_version": "1.0",
            "pass_fail_status": "PASSED",
            "instruction": "汇总状态。",
            "requested_output": "status",
        },
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )


def test_client_refuses_before_network_when_live_calls_are_off():
    registry = load_provider_registry(CONFIG)
    runtime = resolve_provider_runtime(registry, "deepseek_executor", environ={})
    with pytest.raises(ExternalLLMCallsDisabled):
        call_structured(
            runtime=runtime,
            task=_safe_task(),
            output_model=_TinyOutput,
            budget=CallBudget(1),
        )


def test_client_validates_mock_response_and_enforces_call_budget():
    registry = load_provider_registry(CONFIG)
    runtime = resolve_provider_runtime(
        registry,
        "deepseek_executor",
        environ={"COPPER_MAS_LIVE_CALLS": "true", "DEEPSEEK_API_KEY": "local-test-value"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.deepseek.com/chat/completions"
        assert request.headers["Authorization"] == "Bearer local-test-value"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"status": "ok"})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    budget = CallBudget(1)
    result = call_structured(
        runtime=runtime,
        task=_safe_task(),
        output_model=_TinyOutput,
        budget=budget,
        transport=httpx.MockTransport(handler),
    )
    assert result.value.status == "ok"
    assert result.input_tokens == 10
    assert result.output_tokens == 2
    assert result.estimated_cost_cny == pytest.approx((10 * 9.9 + 2 * 29.7) / 1_000_000)
    assert result.cost_status == "ESTIMATED_FROM_REPORTED_TOKENS"
    with pytest.raises(ExternalLLMBudgetExceeded):
        call_structured(
            runtime=runtime,
            task=_safe_task(),
            output_model=_TinyOutput,
            budget=budget,
            transport=httpx.MockTransport(handler),
        )


def test_local_env_loader_returns_names_only_and_budget_overrides(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KIMI_API_KEY=local-secret-value\nCOPPER_MAS_MAX_CALLS_PER_RUN=3\nCOPPER_MAS_MONTHLY_CNY_LIMIT=20\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}
    loaded = load_local_env(env_file, environ=environ)
    assert set(loaded) == {
        "KIMI_API_KEY",
        "COPPER_MAS_MAX_CALLS_PER_RUN",
        "COPPER_MAS_MONTHLY_CNY_LIMIT",
    }
    assert "local-secret-value" not in repr(loaded)
    registry = load_provider_registry(CONFIG)
    budget = resolve_budget_runtime(registry, environ=environ)
    assert budget.max_calls_per_run == 3
    assert budget.monthly_cny_limit == 20


def test_shared_env_loader_can_ignore_unrelated_provider_variables(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UNRELATED_API_KEY=must-not-load\nKIMI_API_KEY=allowed-local-value\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}
    loaded = load_local_env(
        env_file,
        environ=environ,
        ignore_unrelated=True,
    )
    assert loaded == ("KIMI_API_KEY",)
    assert "UNRELATED_API_KEY" not in environ
