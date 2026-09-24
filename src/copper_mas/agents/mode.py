"""A2 确定性工况识别。

只使用预测起点已经可用的过程状态。这里的“开/停”是对记录状态的
描述，不是控制动作，也不推断一期或二期内部独立的一段/二段状态。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from copper_mas.contracts.cards import ProcessModeCardV2
from copper_mas.data.leakage import assert_no_future_information


@dataclass(frozen=True)
class ModeRuleConfig:
    """不依赖目标分布的物理零值规则。"""

    current_on_above: float = 0.0
    companion_on_above: float = 0.0
    lookback_offsets_hours: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12)


@dataclass(frozen=True)
class _CircuitSpec:
    code: str
    current_tag: str
    companion_tags: tuple[str, ...]


_CIRCUITS = (
    _CircuitSpec(
        code="P1_STAGE12",
        current_tag="phase1_stage12_current_ka",
        companion_tags=("phase1_stage12_voltage_v", "phase1_stage12_active_cells"),
    ),
    _CircuitSpec(
        code="P2_STAGE12",
        current_tag="phase2_stage12_current_ka",
        companion_tags=("phase2_stage12_voltage_v", "phase2_stage12_active_cells"),
    ),
    _CircuitSpec(
        code="S3",
        current_tag="stage3_current_a",
        companion_tags=("stage3_voltage_v", "stage3_flow_m3_h"),
    ),
    _CircuitSpec(
        code="S4",
        current_tag="stage4_current_a",
        companion_tags=("stage4_voltage_v", "stage4_flow_m3_h"),
    ),
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _latest_value(
    row: Mapping[str, Any],
    tag: str,
    offsets: tuple[int, ...],
) -> tuple[float | None, str | None]:
    for offset in offsets:
        key = f"{tag}__t_minus_{offset}h"
        value = _number(row.get(key))
        missing = row.get(f"{key}__missing")
        if value is not None and missing not in {True, "1", "true", "True"}:
            return value, key
    return None, None


def _circuit_state(
    row: Mapping[str, Any],
    spec: _CircuitSpec,
    config: ModeRuleConfig,
) -> tuple[str, float, tuple[str, ...], tuple[str, ...]]:
    current, current_key = _latest_value(row, spec.current_tag, config.lookback_offsets_hours)
    evidence: list[str] = []
    warnings: list[str] = []
    if current_key:
        evidence.append(current_key)
    if current is None:
        return "UNKNOWN", 0.0, tuple(evidence), (f"{spec.code}_CURRENT_MISSING",)

    companion_states: list[bool] = []
    for tag in spec.companion_tags:
        value, key = _latest_value(row, tag, config.lookback_offsets_hours)
        if key:
            evidence.append(key)
        if value is not None:
            companion_states.append(value > config.companion_on_above)

    state = "ON" if current > config.current_on_above else "OFF"
    if companion_states:
        corroborated = any(companion_states) if state == "ON" else not any(companion_states)
        confidence = 1.0 if corroborated else 0.6
        if not corroborated:
            warnings.append(f"{spec.code}_SIGNAL_CONFLICT")
    else:
        confidence = 0.75
        warnings.append(f"{spec.code}_COMPANION_MISSING")
    return state, confidence, tuple(evidence), tuple(warnings)


def infer_process_mode(
    *,
    origin_event_id: str,
    decision_at: datetime,
    feature_row: Mapping[str, Any],
    config: ModeRuleConfig | None = None,
) -> ProcessModeCardV2:
    """由 As-Of 特征行生成不可被 LLM 覆盖的工况卡。"""

    assert_no_future_information(feature_row, decision_at=decision_at)
    rule = config or ModeRuleConfig()
    states: list[str] = []
    confidences: list[float] = []
    evidence: list[str] = []
    warnings: list[str] = []
    for spec in _CIRCUITS:
        state, confidence, used, circuit_warnings = _circuit_state(feature_row, spec, rule)
        states.append(f"{spec.code}_{state}")
        confidences.append(confidence)
        evidence.extend(used)
        warnings.extend(circuit_warnings)

    known_confidences = [value for value in confidences if value > 0]
    overall = sum(known_confidences) / len(known_confidences) if known_confidences else 0.0
    return ProcessModeCardV2(
        origin_event_id=origin_event_id,
        decision_at=decision_at,
        mode_code="__".join(states),
        confidence=round(overall, 6),
        evidence_observation_ids=tuple(dict.fromkeys(evidence)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
