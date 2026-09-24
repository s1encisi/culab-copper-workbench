"""对进入预测路径的任意嵌套载荷执行 fail-closed 泄漏检查。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel

FORBIDDEN_EXACT_KEYS = {
    "forward_prediction_flag",
    "target_event_id",
    "target_recorded_at",
    "target_sample_at_assumed",
    "target_source_quality_eligible",
}
FORBIDDEN_PREFIXES = ("target_", "next_", "lead_to_")


class FutureInformationError(ValueError):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value


def find_forbidden_paths(payload: Any, path: str = "$") -> list[str]:
    payload = _plain(payload)
    failures: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_EXACT_KEYS or key_text.startswith(FORBIDDEN_PREFIXES):
                failures.append(child_path)
            failures.extend(find_forbidden_paths(value, child_path))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            failures.extend(find_forbidden_paths(value, f"{path}[{index}]"))
    return failures


def _find_future_availability(payload: Any, decision_at: datetime, path: str = "$") -> list[str]:
    payload = _plain(payload)
    failures: list[str] = []
    if isinstance(payload, Mapping):
        available = payload.get("available_at")
        if isinstance(available, datetime) and available > decision_at:
            failures.append(f"{path}.available_at")
        for key, value in payload.items():
            failures.extend(_find_future_availability(value, decision_at, f"{path}.{key}"))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            failures.extend(_find_future_availability(value, decision_at, f"{path}[{index}]"))
    return failures


def assert_no_future_information(payload: Any, decision_at: datetime | None = None) -> None:
    failures = find_forbidden_paths(payload)
    if decision_at is not None:
        failures.extend(_find_future_availability(payload, decision_at))
    if failures:
        raise FutureInformationError("预测载荷含未来信息: " + ", ".join(sorted(set(failures))))
