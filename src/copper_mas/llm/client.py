"""受白名单、开关和调用预算约束的结构化 Chat Completions 客户端。"""

from __future__ import annotations

import json
import math
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from copper_mas.config.settings import ProviderRuntime, redact_text
from copper_mas.llm.adapters import (
    DeepSeekV4ProAdapter,
    KimiK3Adapter,
    StructuredTask,
)


T = TypeVar("T", bound=BaseModel)


class ExternalLLMCallsDisabled(RuntimeError):
    pass


class ExternalLLMBudgetExceeded(RuntimeError):
    pass


class ExternalLLMResponseError(RuntimeError):
    pass


class CallBudget:
    """单次运行的调用数与人民币成本硬预算。

    网络请求前先按“载荷 UTF-8 字节数 + 最大输出 token”保守预留成本；
    服务端返回 usage 后再以实际 token 数结算。失败请求保留预留额，避免
    在计费状态未知时继续调用。
    """

    def __init__(
        self,
        max_calls: int,
        max_cost_cny: float | None = None,
        *,
        provider_call_limits: dict[str, int] | None = None,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls 必须大于0")
        if max_cost_cny is not None and (
            not math.isfinite(max_cost_cny) or max_cost_cny <= 0
        ):
            raise ValueError("max_cost_cny 必须是大于0的有限数")
        limits = dict(provider_call_limits or {})
        if any(not name or limit <= 0 for name, limit in limits.items()):
            raise ValueError("provider_call_limits 必须使用非空名称和正整数")
        self.max_calls = max_calls
        self.max_cost_cny = max_cost_cny
        self.provider_call_limits = limits
        self.used_calls = 0
        self.used_cost_cny = 0.0
        self._provider_used_calls: dict[str, int] = {}
        self._reservations: dict[int, float] = {}
        self._next_reservation_id = 1

    @property
    def reserved_cost_cny(self) -> float:
        return sum(self._reservations.values())

    @property
    def committed_cost_cny(self) -> float:
        return self.used_cost_cny + self.reserved_cost_cny

    def provider_used_calls(self, provider_name: str) -> int:
        return self._provider_used_calls.get(provider_name, 0)

    def reserve(
        self,
        *,
        provider_name: str = "UNSPECIFIED",
        max_possible_cost_cny: float = 0.0,
    ) -> int:
        if not math.isfinite(max_possible_cost_cny) or max_possible_cost_cny < 0:
            raise ValueError("max_possible_cost_cny 必须是非负有限数")
        if self.used_calls >= self.max_calls:
            raise ExternalLLMBudgetExceeded("本次运行的外部 LLM 调用预算已耗尽")
        provider_limit = self.provider_call_limits.get(provider_name)
        if (
            provider_limit is not None
            and self.provider_used_calls(provider_name) >= provider_limit
        ):
            raise ExternalLLMBudgetExceeded(
                f"{provider_name} 的本次运行调用上限已耗尽"
            )
        projected = self.committed_cost_cny + max_possible_cost_cny
        if self.max_cost_cny is not None and projected > self.max_cost_cny + 1e-12:
            raise ExternalLLMBudgetExceeded(
                "本次运行的人民币成本硬上限不足以覆盖本次请求"
            )
        self.used_calls += 1
        self._provider_used_calls[provider_name] = (
            self.provider_used_calls(provider_name) + 1
        )
        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1
        self._reservations[reservation_id] = max_possible_cost_cny
        return reservation_id

    def settle(self, reservation_id: int, actual_cost_cny: float) -> None:
        if reservation_id not in self._reservations:
            raise ValueError("未知或已结算的成本预留标识")
        if not math.isfinite(actual_cost_cny) or actual_cost_cny < 0:
            raise ValueError("actual_cost_cny 必须是非负有限数")
        reserved = self._reservations.pop(reservation_id)
        self.used_cost_cny += actual_cost_cny
        if actual_cost_cny > reserved + 1e-12:
            raise ExternalLLMBudgetExceeded(
                "服务端报告成本超过请求前保守预留，已停止后续调用"
            )
        if (
            self.max_cost_cny is not None
            and self.committed_cost_cny > self.max_cost_cny + 1e-12
        ):
            raise ExternalLLMBudgetExceeded("本次运行的人民币成本硬上限已超出")


class StructuredCallResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: str
    model: str
    value: T
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_cny: float = Field(default=0.0, ge=0)
    cost_status: str = "NOT_COMPUTED"


def estimate_token_cost_cny(
    runtime: ProviderRuntime, *, input_tokens: int, output_tokens: int
) -> float:
    """依据冻结在 provider 配置中的保守单价估算人民币成本。"""

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token 数不得为负")
    return (
        input_tokens * runtime.input_cny_per_million_tokens
        + output_tokens * runtime.output_cny_per_million_tokens
    ) / 1_000_000.0


def _adapter(runtime: ProviderRuntime) -> KimiK3Adapter | DeepSeekV4ProAdapter:
    if runtime.provider == "moonshot":
        return KimiK3Adapter()
    if runtime.provider == "deepseek":
        return DeepSeekV4ProAdapter()
    raise ValueError(f"不支持的 provider: {runtime.provider}")


def _extract_json_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExternalLLMResponseError("响应缺少 choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ExternalLLMResponseError("响应 content 不是非空 JSON 字符串")
    return content.strip()


def call_structured(
    *,
    runtime: ProviderRuntime,
    task: StructuredTask,
    output_model: type[T],
    budget: CallBudget,
    transport: httpx.BaseTransport | None = None,
    timeout_seconds: float = 60.0,
) -> StructuredCallResult[T]:
    """执行一次严格结构化调用；离线开关关闭时在建连前失败。"""

    if not runtime.live_calls:
        raise ExternalLLMCallsDisabled("COPPER_MAS_LIVE_CALLS=false，未发起网络请求")
    if runtime.base_url is None or runtime.api_key is None:
        raise ExternalLLMCallsDisabled("真实调用配置不完整，未发起网络请求")
    request_payload = _adapter(runtime).build_payload(runtime, task)
    # UTF-8 字节数是不依赖供应商 tokenizer 的保守输入 token 上界。
    conservative_input_tokens = len(
        json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    reserved_cost_cny = estimate_token_cost_cny(
        runtime,
        input_tokens=conservative_input_tokens,
        output_tokens=runtime.max_completion_tokens,
    )
    reservation_id = budget.reserve(
        provider_name=runtime.name,
        max_possible_cost_cny=reserved_cost_cny,
    )
    endpoint = runtime.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {runtime.api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(transport=transport, timeout=timeout_seconds) as client:
            response = client.post(endpoint, headers=headers, json=request_payload)
            response.raise_for_status()
            response_payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        safe = redact_text(str(exc))
        raise ExternalLLMResponseError(f"结构化调用失败: {safe}") from exc

    usage = response_payload.get("usage") or {}
    try:
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError) as exc:
        raise ExternalLLMResponseError("响应 usage token 计数无效") from exc
    if input_tokens < 0 or output_tokens < 0:
        raise ExternalLLMResponseError("响应 usage token 计数不得为负")
    usage_reported = bool(usage) and (
        "prompt_tokens" in usage or "completion_tokens" in usage
    )
    if usage_reported:
        estimated_cost_cny = estimate_token_cost_cny(
            runtime,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        budget.settle(reservation_id, estimated_cost_cny)
        cost_status = "ESTIMATED_FROM_REPORTED_TOKENS"
    else:
        # 计费状态未知时不释放事前预留额。
        estimated_cost_cny = reserved_cost_cny
        cost_status = "CONSERVATIVE_UPPER_BOUND_NO_USAGE"

    content = _extract_json_content(response_payload)
    try:
        value = output_model.model_validate_json(content)
    except ValueError as exc:
        raise ExternalLLMResponseError("响应 JSON 未通过本地 Pydantic 合同") from exc
    return StructuredCallResult[T](
        provider_name=runtime.name,
        model=runtime.model,
        value=value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_cny=estimated_cost_cny,
        cost_status=cost_status,
    )
