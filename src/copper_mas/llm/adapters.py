"""Kimi K3 与 DeepSeek V4 Pro 的纯函数载荷适配器。"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from copper_mas.config.settings import ProviderRuntime, SECRET_PATTERN
from copper_mas.data.leakage import assert_no_future_information


class SafeLLMEnvelope(BaseModel):
    """允许跨出本机的最小元数据包；不接受事件行或观测列表。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    schema_version: str = Field(min_length=1, max_length=32)
    pass_fail_status: str = Field(min_length=1, max_length=32)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=64)
    bounded_aggregate_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=64
    )
    instruction: str = Field(min_length=1, max_length=1000)
    requested_output: str = Field(min_length=1, max_length=500)

    @field_validator("artifact_ids", "reason_codes")
    @classmethod
    def validate_string_sequences(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value or len(value) > 128:
                raise ValueError("元数据标识/原因码必须为1—128字符")
            if SECRET_PATTERN.search(value):
                raise ValueError("元数据包疑似包含密钥")
        return values

    @model_validator(mode="after")
    def validate_scalar_metadata(self) -> "SafeLLMEnvelope":
        for key, value in self.bounded_aggregate_metadata.items():
            if not key or len(key) > 128:
                raise ValueError("聚合元数据键必须为1—128字符")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("聚合元数据不得包含非有限数")
            if isinstance(value, str) and (len(value) > 256 or SECRET_PATTERN.search(value)):
                raise ValueError("聚合元数据字符串过长或疑似包含密钥")
        for text in (self.instruction, self.requested_output, self.pass_fail_status):
            if SECRET_PATTERN.search(text):
                raise ValueError("LLM 元数据包疑似包含密钥")
        return self


class StructuredTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    system_prompt: str = Field(min_length=1, max_length=4000)
    user_payload: SafeLLMEnvelope
    output_schema: dict[str, Any]

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: str) -> str:
        if SECRET_PATTERN.search(value):
            raise ValueError("system_prompt 疑似包含密钥")
        return value


def _messages(task: StructuredTask) -> list[dict[str, str]]:
    import json

    return [
        {"role": "system", "content": task.system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                task.user_payload.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


class KimiK3Adapter:
    def build_payload(self, runtime: ProviderRuntime, task: StructuredTask) -> dict[str, Any]:
        assert_no_future_information(task.user_payload)
        return {
            "model": runtime.model,
            "messages": _messages(task),
            "reasoning_effort": runtime.reasoning_effort,
            "max_completion_tokens": runtime.max_completion_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": task.task_name,
                    "strict": True,
                    "schema": task.output_schema,
                },
            },
        }


class DeepSeekV4ProAdapter:
    def build_payload(self, runtime: ProviderRuntime, task: StructuredTask) -> dict[str, Any]:
        assert_no_future_information(task.user_payload)
        return {
            "model": runtime.model,
            "messages": _messages(task),
            "reasoning_effort": runtime.reasoning_effort,
            "thinking": {"type": "enabled"},
            "max_tokens": runtime.max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
