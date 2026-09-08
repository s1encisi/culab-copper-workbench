"""只从环境变量解析密钥，并对白名单 URL 和调用开关做强校验。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr


SECRET_PATTERN = re.compile(r"(?:sk-|Bearer\s+)[A-Za-z0-9_\-]{12,}", re.IGNORECASE)


class ConfigurationError(ValueError):
    pass


class ProviderPricingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_currency: str
    source_input_per_million_tokens: float = Field(ge=0)
    source_output_per_million_tokens: float = Field(ge=0)
    usd_to_cny_upper_bound: float | None = Field(default=None, gt=0)
    input_cny_per_million_tokens: float = Field(ge=0)
    output_cny_per_million_tokens: float = Field(ge=0)
    basis: str = Field(min_length=1)


class ProviderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model_env: str
    model_default: str
    api_key_env: str
    base_url_env: str
    base_url_default: str | None = None
    allowed_base_urls: tuple[str, ...]
    api_style: str
    structured_output: str
    reasoning_effort: str
    max_completion_tokens: int = Field(gt=0)
    max_calls_per_run: int = Field(gt=0)
    pricing: ProviderPricingSpec
    thinking: str | None = None


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_calls_per_run: int = Field(gt=0)
    max_cny_per_run: float = Field(gt=0)
    max_repair_attempts: int = Field(ge=0)
    max_tool_rounds: int = Field(ge=0)
    max_wall_seconds: int = Field(gt=0)
    monthly_cny_limit: float | None = Field(default=None, gt=0)


class ProviderRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    live_calls_default: bool
    data_policy: dict[str, Any]
    providers: dict[str, ProviderSpec]
    budgets: BudgetSpec


class ProviderRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    live_calls: bool
    provider: str
    model: str
    base_url: str | None
    api_key: SecretStr | None
    reasoning_effort: str
    max_completion_tokens: int
    max_calls_per_run: int
    input_cny_per_million_tokens: float = Field(default=0.0, ge=0)
    output_cny_per_million_tokens: float = Field(default=0.0, ge=0)
    pricing_basis: str = "UNCONFIGURED"


class BudgetRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_calls_per_run: int = Field(gt=0)
    max_cny_per_run: float = Field(gt=0)
    max_kimi_calls_per_run: int = Field(gt=0)
    max_deepseek_calls_per_run: int = Field(gt=0)
    max_repair_attempts: int = Field(ge=0)
    max_tool_rounds: int = Field(ge=0)
    max_wall_seconds: int = Field(gt=0)
    monthly_cny_limit: float | None = Field(default=None, gt=0)


ALLOWED_LOCAL_ENV_PREFIXES = ("COPPER_MAS_", "KIMI_", "DEEPSEEK_")


def load_local_env(
    path: str | Path,
    *,
    environ: dict[str, str] | None = None,
    override: bool = False,
    ignore_unrelated: bool = False,
) -> tuple[str, ...]:
    """静默加载本机 .env，只返回白名单变量名，绝不返回或打印变量值。"""

    target = os.environ if environ is None else environ
    values = dotenv_values(Path(path))
    unexpected = sorted(
        key for key in values if not key.startswith(ALLOWED_LOCAL_ENV_PREFIXES)
    )
    if unexpected and not ignore_unrelated:
        raise ConfigurationError(f".env 含未允许的变量名: {unexpected}")
    loaded: list[str] = []
    for key, value in values.items():
        if not key.startswith(ALLOWED_LOCAL_ENV_PREFIXES):
            continue
        if value is None:
            continue
        if override or key not in target:
            target[key] = value
            loaded.append(key)
    return tuple(sorted(loaded))


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须是整数") from exc


def resolve_budget_runtime(
    registry: ProviderRegistry,
    *,
    environ: dict[str, str] | None = None,
) -> BudgetRuntime:
    env = os.environ if environ is None else environ
    monthly_raw = env.get("COPPER_MAS_MONTHLY_CNY_LIMIT", "").strip()
    try:
        monthly = float(monthly_raw) if monthly_raw else registry.budgets.monthly_cny_limit
    except ValueError as exc:
        raise ConfigurationError("COPPER_MAS_MONTHLY_CNY_LIMIT 必须是数值") from exc
    return BudgetRuntime(
        max_calls_per_run=_env_int(
            env, "COPPER_MAS_MAX_CALLS_PER_RUN", registry.budgets.max_calls_per_run
        ),
        max_cny_per_run=registry.budgets.max_cny_per_run,
        max_kimi_calls_per_run=_env_int(
            env,
            "COPPER_MAS_MAX_KIMI_CALLS_PER_RUN",
            registry.providers["kimi_orchestrator"].max_calls_per_run,
        ),
        max_deepseek_calls_per_run=_env_int(
            env,
            "COPPER_MAS_MAX_DEEPSEEK_CALLS_PER_RUN",
            registry.providers["deepseek_executor"].max_calls_per_run,
        ),
        max_repair_attempts=_env_int(
            env, "COPPER_MAS_MAX_REPAIR_ATTEMPTS", registry.budgets.max_repair_attempts
        ),
        max_tool_rounds=_env_int(
            env, "COPPER_MAS_MAX_TOOL_ROUNDS", registry.budgets.max_tool_rounds
        ),
        max_wall_seconds=_env_int(
            env, "COPPER_MAS_MAX_WALL_SECONDS", registry.budgets.max_wall_seconds
        ),
        monthly_cny_limit=monthly,
    )


def _walk_for_inline_secrets(value: Any, path: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(_walk_for_inline_secrets(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_for_inline_secrets(child, f"{path}[{index}]"))
    elif isinstance(value, str) and SECRET_PATTERN.search(value):
        failures.append(path)
    return failures


def load_provider_registry(path: str | Path) -> ProviderRegistry:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    inline_secrets = _walk_for_inline_secrets(raw)
    if inline_secrets:
        raise ConfigurationError("配置文件疑似包含明文密钥: " + ", ".join(inline_secrets))
    return ProviderRegistry.model_validate(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} 必须是 true/false")


def resolve_provider_runtime(
    registry: ProviderRegistry,
    provider_name: str,
    *,
    environ: dict[str, str] | None = None,
) -> ProviderRuntime:
    env = os.environ if environ is None else environ
    if provider_name not in registry.providers:
        raise ConfigurationError(f"未知 provider: {provider_name}")
    spec = registry.providers[provider_name]
    live_raw = env.get("COPPER_MAS_LIVE_CALLS")
    if live_raw is None:
        live_calls = registry.live_calls_default
    else:
        normalized = live_raw.strip().lower()
        if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ConfigurationError("COPPER_MAS_LIVE_CALLS 必须是 true/false")
        live_calls = normalized in {"1", "true", "yes", "on"}

    model = env.get(spec.model_env, spec.model_default).strip()
    base_url = env.get(spec.base_url_env, spec.base_url_default or "").strip() or None
    key_text = env.get(spec.api_key_env, "").strip()

    if base_url is not None and base_url.rstrip("/") not in {url.rstrip("/") for url in spec.allowed_base_urls}:
        raise ConfigurationError(f"{provider_name} BASE_URL 不在白名单")
    if live_calls and base_url is None:
        raise ConfigurationError(f"{provider_name} 启用真实调用前必须明确 BASE_URL")
    if live_calls and not key_text:
        raise ConfigurationError(f"{provider_name} 启用真实调用前必须在环境变量 {spec.api_key_env} 配置密钥")

    return ProviderRuntime(
        name=provider_name,
        live_calls=live_calls,
        provider=spec.provider,
        model=model,
        base_url=base_url,
        api_key=SecretStr(key_text) if key_text else None,
        reasoning_effort=spec.reasoning_effort,
        max_completion_tokens=spec.max_completion_tokens,
        max_calls_per_run=spec.max_calls_per_run,
        input_cny_per_million_tokens=spec.pricing.input_cny_per_million_tokens,
        output_cny_per_million_tokens=spec.pricing.output_cny_per_million_tokens,
        pricing_basis=spec.pricing.basis,
    )


def redact_text(text: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED_SECRET]", text)
