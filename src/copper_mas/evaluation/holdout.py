"""2026 外部时间留出的一次性释放闸门。"""

from __future__ import annotations

from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class ExternalHoldoutLocked(PermissionError):
    pass


class RequiredFreezesV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timing_contract_v2_path: str | None = None
    timing_contract_v2_sha256: str
    core_feature_contract_path: str | None = None
    core_feature_contract_sha256: str
    model_selection_protocol_path: str
    model_selection_protocol_sha256: str | None = None
    selected_model_manifest_path: str | None
    selected_model_manifest_sha256: str | None
    agent_graph_path: str
    frozen_agent_graph_sha256: str | None
    human_review_status: str
    human_review_decision_path: str | None = None
    human_review_decision_sha256: str | None = None
    human_review_workbook_sha256: str | None = None
    external_evaluation_protocol_path: str | None = None
    external_evaluation_protocol_sha256: str | None = None
    external_evaluator_path: str | None = None
    external_evaluator_sha256: str | None = None
    external_runtime_manifest_path: str | None = None
    external_runtime_manifest_sha256: str | None = None


class ExternalReleaseGateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: Literal["EXTERNAL_2026_RELEASE_GATE_V1"]
    status: Literal[
        "LOCKED_MODEL_AND_GRAPH_NOT_FROZEN",
        "LOCKED_HUMAN_REVIEW_PENDING",
        "RELEASED_FOR_SINGLE_EVALUATION",
        "RUNNING_EXTERNAL_EVALUATION",
        "CONSUMED_EXTERNAL_EVALUATION",
        "FAILED_CLOSED_EXTERNAL_EVALUATION",
    ]
    created_on: date
    claim_boundary: str
    required_freezes: RequiredFreezesV1
    safe_pre_freeze_access: tuple[str, ...]
    forbidden_pre_freeze_access: tuple[str, ...]
    release_rule: str
    single_evaluation_only: bool
    post_evaluation_status: str
    release_evidence_path: str | None = None
    release_evidence_sha256: str | None = None


def load_external_release_gate(path: str | Path) -> ExternalReleaseGateV1:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExternalReleaseGateV1.model_validate(payload)


def _normalized(value: str | Path) -> str:
    return str(PurePosixPath(str(value).replace("\\", "/"))).lstrip("./")


def assert_pre_freeze_path_allowed(gate: ExternalReleaseGateV1, relative_path: str | Path) -> None:
    normalized = _normalized(relative_path)
    forbidden = {_normalized(item) for item in gate.forbidden_pre_freeze_access}
    if gate.status != "RELEASED_FOR_SINGLE_EVALUATION" and normalized in forbidden:
        raise ExternalHoldoutLocked(f"外部留出仍锁定，禁止读取: {normalized}")


def assert_external_evaluation_released(gate: ExternalReleaseGateV1) -> None:
    if gate.status != "RELEASED_FOR_SINGLE_EVALUATION":
        raise ExternalHoldoutLocked(f"外部评价未释放，当前状态: {gate.status}")
    freeze = gate.required_freezes
    missing = []
    if not freeze.selected_model_manifest_path:
        missing.append("selected_model_manifest_path")
    if not freeze.selected_model_manifest_sha256:
        missing.append("selected_model_manifest_sha256")
    if not freeze.frozen_agent_graph_sha256:
        missing.append("frozen_agent_graph_sha256")
    if freeze.human_review_status != "COMPLETED_AND_ACCEPTED":
        missing.append("human_review_status")
    if not freeze.human_review_decision_path:
        missing.append("human_review_decision_path")
    if not freeze.human_review_decision_sha256:
        missing.append("human_review_decision_sha256")
    if not freeze.human_review_workbook_sha256:
        missing.append("human_review_workbook_sha256")
    if not freeze.external_evaluation_protocol_path:
        missing.append("external_evaluation_protocol_path")
    if not freeze.external_evaluation_protocol_sha256:
        missing.append("external_evaluation_protocol_sha256")
    if not freeze.external_evaluator_path:
        missing.append("external_evaluator_path")
    if not freeze.external_evaluator_sha256:
        missing.append("external_evaluator_sha256")
    if not freeze.timing_contract_v2_path:
        missing.append("timing_contract_v2_path")
    if not freeze.core_feature_contract_path:
        missing.append("core_feature_contract_path")
    if not freeze.model_selection_protocol_sha256:
        missing.append("model_selection_protocol_sha256")
    if not freeze.external_runtime_manifest_path:
        missing.append("external_runtime_manifest_path")
    if not freeze.external_runtime_manifest_sha256:
        missing.append("external_runtime_manifest_sha256")
    if gate.single_evaluation_only is not True:
        missing.append("single_evaluation_only")
    if gate.post_evaluation_status != "CONSUMED_EXTERNAL_EVALUATION":
        missing.append("post_evaluation_status")
    if not gate.release_evidence_path:
        missing.append("release_evidence_path")
    if not gate.release_evidence_sha256:
        missing.append("release_evidence_sha256")
    if missing:
        raise ExternalHoldoutLocked("外部评价释放条件不完整: " + ", ".join(missing))
