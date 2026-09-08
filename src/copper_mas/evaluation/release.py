"""在不读取 sealed 正文的前提下，原子释放一次性 2026 外评门。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from copper_mas.evaluation.external_once import header_only_real_preflight
from copper_mas.evaluation.holdout import (
    ExternalReleaseGateV1,
    load_external_release_gate,
)
from copper_mas.evaluation.transaction import (
    ExternalEvaluationTransactionError,
    _atomic_write_bytes,
    _atomic_write_json,
    _create_exclusive_json,
    sha256_file,
    verify_release_integrity,
    verify_runtime_freeze_manifest,
)


class ExternalReleasePreparationError(RuntimeError):
    pass


def _require_p0_current(
    *, project_root: Path, gate_path: Path, report_path: Path
) -> tuple[dict[str, Any], str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ExternalReleasePreparationError("P0 审计未通过")
    if any(item.get("passed") is not True for item in report.get("checks", [])):
        raise ExternalReleasePreparationError("P0 存在未通过检查")
    hashes = report.get("sha256") or {}
    actual_gate_hash = sha256_file(gate_path)
    if hashes.get("external_release_gate") != actual_gate_hash:
        raise ExternalReleasePreparationError("P0 报告未绑定当前释放门")
    required_current = {
        "v2_contract": project_root.parent
        / "01_时间合同与A1数据准备/预测任务与时间对齐合同_V2.md",
        "core_feature_contract": project_root
        / "configs/contracts/core_features_v2.yaml",
        "model_selection_protocol": project_root
        / "contracts/frozen/model_selection_protocol_v1.yaml",
        "selected_model_manifest": project_root
        / "contracts/frozen/selected_model_manifest_v1.json",
        "agent_graph": project_root / "configs/agents/graph_v1.yaml",
        "manual_review_decision": project_root
        / "artifacts/p1/review/manual_review_decision_v2.json",
        "manual_review_workbook": project_root
        / "artifacts/p1/review/铜电解电积P1人工抽核30条_V2.xlsx",
        "external_evaluation_protocol": project_root
        / "contracts/frozen/external_evaluation_protocol_v1.yaml",
        "external_evaluator": project_root
        / "src/copper_mas/evaluation/external_once.py",
        "external_runtime_manifest": project_root
        / "contracts/frozen/external_runtime_freeze_manifest_v1.json",
    }
    for key, path in required_current.items():
        if not path.is_file() or hashes.get(key) != sha256_file(path):
            raise ExternalReleasePreparationError(f"P0 报告已过期或缺少: {key}")
    return report, actual_gate_hash


def release_external_gate_atomically(
    *,
    project_root: str | Path,
    gate_path: str | Path,
    p0_report_path: str | Path,
    preflight_report_path: str | Path,
    review_decision_path: str | Path,
    protocol_path: str | Path,
    external_dir: str | Path,
    claim_path: str | Path,
    formal_output_dir: str | Path,
    release_evidence_path: str | Path,
) -> dict[str, Any]:
    """校验全部前置证据后，以 CAS 方式将 LOCKED 转为 RELEASED。"""

    root = Path(project_root).resolve()
    gate_file = Path(gate_path).resolve()
    claim_file = Path(claim_path).resolve()
    output_dir = Path(formal_output_dir).resolve()
    evidence_file = Path(release_evidence_path).resolve()
    preflight_file = Path(preflight_report_path).resolve()
    p0_file = Path(p0_report_path).resolve()
    gate = load_external_release_gate(gate_file)
    if gate.status != "LOCKED_HUMAN_REVIEW_PENDING":
        raise ExternalReleasePreparationError(
            f"只能从 LOCKED_HUMAN_REVIEW_PENDING 释放: {gate.status}"
        )
    if claim_file.exists():
        raise ExternalReleasePreparationError("已存在 claim，禁止释放")
    if output_dir.exists():
        raise ExternalReleasePreparationError("已存在正式外评输出，禁止释放")
    if evidence_file.exists():
        raise ExternalReleasePreparationError("释放证据已存在，禁止重复释放")

    p0_report, gate_hash_before = _require_p0_current(
        project_root=root, gate_path=gate_file, report_path=p0_file
    )
    preflight = header_only_real_preflight(
        project_root=root,
        gate_path=gate_file,
        review_decision_path=review_decision_path,
        protocol_path=protocol_path,
        external_dir=external_dir,
    )
    if preflight.get("ready_for_release_after_gate_binding") is not True:
        raise ExternalReleasePreparationError("header-only 预检未达到释放条件")
    if preflight.get("sealed_outcome_rows_read") != 0:
        raise ExternalReleasePreparationError("header-only 预检读取了 sealed 正文")
    if preflight.get("formal_evaluation_started") is not False:
        raise ExternalReleasePreparationError("header-only 预检意外启动了正式外评")
    _atomic_write_json(preflight_file, preflight)

    freezes = gate.required_freezes
    runtime = verify_runtime_freeze_manifest(
        project_root=root,
        manifest_path=freezes.external_runtime_manifest_path or "",
        expected_sha256=freezes.external_runtime_manifest_sha256 or "",
    )
    evidence_payload = {
        "evidence_id": "EXTERNAL_2026_RELEASE_EVIDENCE_V1",
        "schema_version": "1.0",
        "authorized_release": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_status_before": gate.status,
        "gate_status_authorized_after": "RELEASED_FOR_SINGLE_EVALUATION",
        "gate_sha256_before_release": gate_hash_before,
        "p0_report_path": str(p0_file),
        "p0_report_sha256": sha256_file(p0_file),
        "p0_audit_id": p0_report.get("audit_id"),
        "header_only_preflight_path": str(preflight_file),
        "header_only_preflight_sha256": sha256_file(preflight_file),
        "sealed_outcome_rows_read": 0,
        "sealed_outcome_actual_sha256_computed": False,
        "formal_evaluation_started": False,
        "claim_absent": True,
        "formal_output_absent": True,
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "runtime_verified_file_count": runtime["verified_file_count"],
        "selected_model_manifest_sha256": freezes.selected_model_manifest_sha256,
        "agent_graph_sha256": freezes.frozen_agent_graph_sha256,
        "external_evaluation_protocol_sha256": freezes.external_evaluation_protocol_sha256,
        "external_evaluator_sha256": freezes.external_evaluator_sha256,
    }
    _create_exclusive_json(evidence_file, evidence_payload)
    evidence_hash = sha256_file(evidence_file)

    try:
        evidence_relative = evidence_file.relative_to(root).as_posix()
    except ValueError as exc:
        raise ExternalReleasePreparationError(
            "释放证据必须位于项目根目录内"
        ) from exc
    candidate_payload = gate.model_dump(mode="json")
    candidate_payload["status"] = "RELEASED_FOR_SINGLE_EVALUATION"
    candidate_payload["release_evidence_path"] = evidence_relative
    candidate_payload["release_evidence_sha256"] = evidence_hash
    candidate = ExternalReleaseGateV1.model_validate(candidate_payload)
    verify_release_integrity(
        project_root=root,
        gate=candidate,
        review_decision_path=review_decision_path,
    )
    if sha256_file(gate_file) != gate_hash_before:
        raise ExternalEvaluationTransactionError(
            "释放门在释放前发生了并发修改"
        )
    encoded = yaml.safe_dump(
        candidate.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).encode("utf-8")
    _atomic_write_bytes(gate_file, encoded)
    result = dict(evidence_payload)
    result["release_evidence_sha256"] = evidence_hash
    result["gate_sha256_after_release"] = sha256_file(gate_file)
    result["gate_status_after"] = load_external_release_gate(gate_file).status
    return result
