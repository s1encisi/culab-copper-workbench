"""2026 外部评价的一次性门禁事务。

这个模块只管理释放、独占 claim、阶段记录与最终消费。它不读取
封存真值；真值读取必须在 ``OUTCOME_ACCESS_STARTED`` 阶段之后由评价器执行。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as distribution_version
import json
import os
from pathlib import Path
import platform
from typing import Any
from uuid import uuid4

import yaml

from copper_mas.evaluation.holdout import (
    ExternalHoldoutLocked,
    ExternalReleaseGateV1,
    assert_external_evaluation_released,
    load_external_release_gate,
)


class ExternalEvaluationTransactionError(RuntimeError):
    """外部评价事务无法安全继续。"""


REQUIRED_RUNTIME_PATHS = frozenset(
    {
        "src/copper_mas/evaluation/external_once.py",
        "src/copper_mas/evaluation/transaction.py",
        "src/copper_mas/evaluation/holdout.py",
        "src/copper_mas/models/predictors.py",
        "src/copper_mas/agents/runtime.py",
        "src/copper_mas/agents/mode.py",
        "src/copper_mas/data/leakage.py",
        "src/copper_mas/contracts/cards.py",
        "src/copper_mas/contracts/runtime.py",
        "scripts/run_external_2026_once_v1.py",
        "scripts/release_external_2026_gate_v1.py",
        "pyproject.toml",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _create_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        raise ExternalEvaluationTransactionError(
            f"一次性 claim 已存在，禁止重复评价: {path}"
        ) from exc
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_claim(path: Path, run_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExternalEvaluationTransactionError("找不到一次性 claim") from exc
    if payload.get("run_id") != run_id:
        raise ExternalEvaluationTransactionError("claim run_id 不匹配")
    return payload


def _resolve_project_file(project_root: Path, value: str | Path) -> Path:
    root = project_root.resolve()
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ExternalEvaluationTransactionError(
            f"冻结工件路径超出项目根目录: {value}"
        ) from exc
    if not path.is_file():
        raise ExternalEvaluationTransactionError(f"冻结工件不存在: {path}")
    return path


def _resolve_workspace_file(project_root: Path, value: str | Path) -> Path:
    """允许时间合同位于工程目录之外，但仍必须位于共享 workspace 内。"""

    project = project_root.resolve()
    workspace = project.parents[1]
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (project / raw).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ExternalEvaluationTransactionError(
            f"冻结工件路径超出 workspace: {value}"
        ) from exc
    if not path.is_file():
        raise ExternalEvaluationTransactionError(f"冻结工件不存在: {path}")
    return path


def verify_runtime_freeze_manifest(
    *, project_root: str | Path, manifest_path: str | Path, expected_sha256: str
) -> dict[str, Any]:
    """逐文件校验外评运行时冻结清单与 Python 依赖版本。"""

    root = Path(project_root).resolve()
    manifest_file = _resolve_project_file(root, manifest_path)
    manifest_hash = _verify_hash(
        manifest_file, expected_sha256, "external_runtime_manifest"
    )
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if payload.get("manifest_id") != "EXTERNAL_2026_RUNTIME_FREEZE_MANIFEST_V1":
        raise ExternalEvaluationTransactionError("外评运行时清单 ID 不匹配")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ExternalEvaluationTransactionError("外评运行时清单缺少 files")
    paths = [str(item.get("path") or "").replace("\\", "/") for item in files]
    if len(paths) != len(set(paths)):
        raise ExternalEvaluationTransactionError("外评运行时清单路径不唯一")
    missing_required = sorted(REQUIRED_RUNTIME_PATHS - set(paths))
    if missing_required:
        raise ExternalEvaluationTransactionError(
            f"外评运行时清单缺少必要依赖: {missing_required}"
        )
    verified: dict[str, str] = {}
    for item, relative in zip(files, paths, strict=True):
        path = _resolve_project_file(root, relative)
        actual = _verify_hash(path, str(item.get("sha256") or ""), relative)
        if int(item.get("size_bytes", -1)) != path.stat().st_size:
            raise ExternalEvaluationTransactionError(
                f"外评运行时文件大小不匹配: {relative}"
            )
        verified[relative] = actual

    expected_versions = payload.get("runtime_versions") or {}
    actual_versions = {
        "python": platform.python_version(),
        "numpy": distribution_version("numpy"),
        "pandas": distribution_version("pandas"),
        "pydantic": distribution_version("pydantic"),
        "PyYAML": distribution_version("PyYAML"),
        "scikit-learn": distribution_version("scikit-learn"),
    }
    if expected_versions != actual_versions:
        raise ExternalEvaluationTransactionError(
            f"外评运行环境版本不匹配: {actual_versions} != {expected_versions}"
        )
    return {
        "manifest_sha256": manifest_hash,
        "verified_file_count": len(verified),
        "runtime_versions": actual_versions,
    }


def _verify_hash(path: Path, expected: str | None, label: str) -> str:
    if not expected:
        raise ExternalEvaluationTransactionError(f"释放门缺少 {label} SHA-256")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ExternalEvaluationTransactionError(f"{label} SHA-256 不匹配")
    return actual


def verify_release_integrity(
    *,
    project_root: str | Path,
    gate: ExternalReleaseGateV1,
    review_decision_path: str | Path,
) -> dict[str, Any]:
    """在创建 claim 前校验全部冻结证据，不读封存真值。"""

    assert_external_evaluation_released(gate)
    root = Path(project_root).resolve()
    freezes = gate.required_freezes

    timing = _resolve_workspace_file(root, freezes.timing_contract_v2_path or "")
    core_contract = _resolve_project_file(
        root, freezes.core_feature_contract_path or ""
    )
    selection_protocol = _resolve_project_file(
        root, freezes.model_selection_protocol_path
    )
    timing_hash = _verify_hash(
        timing, freezes.timing_contract_v2_sha256, "timing_contract_v2"
    )
    core_hash = _verify_hash(
        core_contract,
        freezes.core_feature_contract_sha256,
        "core_feature_contract",
    )
    selection_protocol_hash = _verify_hash(
        selection_protocol,
        freezes.model_selection_protocol_sha256,
        "model_selection_protocol",
    )

    selected = _resolve_project_file(root, freezes.selected_model_manifest_path or "")
    graph = _resolve_project_file(root, freezes.agent_graph_path)
    selected_hash = _verify_hash(
        selected, freezes.selected_model_manifest_sha256, "selected_model_manifest"
    )
    graph_hash = _verify_hash(graph, freezes.frozen_agent_graph_sha256, "agent_graph")

    decision = _resolve_project_file(root, review_decision_path)
    decision_payload = json.loads(decision.read_text(encoding="utf-8"))
    accepted = (
        decision_payload.get("completed") is True
        and decision_payload.get("accepted_for_external_release") is True
        and decision_payload.get("completion_status") == "COMPLETED_AND_ACCEPTED"
    )
    if not accepted:
        raise ExternalEvaluationTransactionError("人工抽核 decision 未通过")
    decision_hash = sha256_file(decision)
    if (
        freezes.human_review_decision_path
        and _resolve_project_file(root, freezes.human_review_decision_path) != decision
    ):
        raise ExternalEvaluationTransactionError("人工抽核 decision 路径与释放门不一致")
    if (
        freezes.human_review_decision_sha256
        and decision_hash.lower() != freezes.human_review_decision_sha256.lower()
    ):
        raise ExternalEvaluationTransactionError("人工抽核 decision SHA-256 不匹配")

    workbook_value = str(decision_payload.get("source_workbook") or "")
    workbook = _resolve_project_file(root, workbook_value)
    workbook_hash = sha256_file(workbook)
    declared_workbook_hash = str(
        decision_payload.get("source_workbook_sha256") or ""
    )
    if not declared_workbook_hash or workbook_hash.lower() != declared_workbook_hash.lower():
        raise ExternalEvaluationTransactionError("人工抽核工作簿 SHA-256 与 decision 不一致")
    if (
        freezes.human_review_workbook_sha256
        and workbook_hash.lower() != freezes.human_review_workbook_sha256.lower()
    ):
        raise ExternalEvaluationTransactionError("人工抽核工作簿 SHA-256 与释放门不一致")

    protocol_hash: str | None = None
    if freezes.external_evaluation_protocol_path:
        protocol = _resolve_project_file(
            root, freezes.external_evaluation_protocol_path
        )
        protocol_hash = _verify_hash(
            protocol,
            freezes.external_evaluation_protocol_sha256,
            "external_evaluation_protocol",
        )

    evaluator_hash: str | None = None
    if freezes.external_evaluator_path:
        evaluator = _resolve_project_file(root, freezes.external_evaluator_path)
        evaluator_hash = _verify_hash(
            evaluator,
            freezes.external_evaluator_sha256,
            "external_evaluator",
        )

    runtime_evidence = verify_runtime_freeze_manifest(
        project_root=root,
        manifest_path=freezes.external_runtime_manifest_path or "",
        expected_sha256=freezes.external_runtime_manifest_sha256 or "",
    )

    release_evidence = _resolve_project_file(root, gate.release_evidence_path or "")
    release_evidence_hash = _verify_hash(
        release_evidence,
        gate.release_evidence_sha256,
        "external_release_evidence",
    )
    release_payload = json.loads(release_evidence.read_text(encoding="utf-8"))
    if release_payload.get("authorized_release") is not True:
        raise ExternalEvaluationTransactionError("外评释放证据未授权")
    if release_payload.get("sealed_outcome_rows_read") != 0:
        raise ExternalEvaluationTransactionError("释放证据显示 sealed 正文曾被读取")
    if release_payload.get("claim_absent") is not True:
        raise ExternalEvaluationTransactionError("释放证据未确认 claim 不存在")
    if release_payload.get("formal_output_absent") is not True:
        raise ExternalEvaluationTransactionError("释放证据未确认正式输出不存在")

    return {
        "timing_contract_v2_sha256": timing_hash,
        "core_feature_contract_sha256": core_hash,
        "model_selection_protocol_sha256": selection_protocol_hash,
        "selected_model_manifest_sha256": selected_hash,
        "agent_graph_sha256": graph_hash,
        "human_review_decision_sha256": decision_hash,
        "human_review_workbook_sha256": workbook_hash,
        "external_evaluation_protocol_sha256": protocol_hash,
        "external_evaluator_sha256": evaluator_hash,
        "external_runtime_manifest_sha256": runtime_evidence["manifest_sha256"],
        "external_runtime_verified_file_count": runtime_evidence[
            "verified_file_count"
        ],
        "external_release_evidence_sha256": release_evidence_hash,
    }


def _replace_gate_status(
    *,
    gate_path: Path,
    expected_status: str,
    new_status: str,
    expected_sha256: str | None = None,
) -> str:
    if expected_sha256 and sha256_file(gate_path).lower() != expected_sha256.lower():
        raise ExternalEvaluationTransactionError("释放门在事务期间发生了并发修改")
    gate = load_external_release_gate(gate_path)
    if gate.status != expected_status:
        raise ExternalEvaluationTransactionError(
            f"释放门状态竞态: {gate.status} != {expected_status}"
        )
    payload = gate.model_dump(mode="json")
    payload["status"] = new_status
    encoded = yaml.safe_dump(
        payload, allow_unicode=True, sort_keys=False, default_flow_style=False
    ).encode("utf-8")
    _atomic_write_bytes(gate_path, encoded)
    return sha256_file(gate_path)


def claim_external_evaluation(
    *,
    project_root: str | Path,
    gate_path: str | Path,
    claim_path: str | Path,
    review_decision_path: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """原子抢占唯一外部评价，并将门从 RELEASED 转为 RUNNING。"""

    gate_file = Path(gate_path).resolve()
    claim_file = Path(claim_path).resolve()
    gate = load_external_release_gate(gate_file)
    evidence = verify_release_integrity(
        project_root=project_root,
        gate=gate,
        review_decision_path=review_decision_path,
    )
    gate_hash_before = sha256_file(gate_file)
    claim = {
        "schema_version": "1.0",
        "transaction_id": "EXTERNAL_2026_SINGLE_EVALUATION_TRANSACTION_V1",
        "run_id": run_id,
        "status": "CLAIMING",
        "phase": "PRE_OUTCOME",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "gate_path": str(gate_file),
        "gate_sha256_before_claim": gate_hash_before,
        "gate_sha256_running": None,
        "outcome_access_started": False,
        "predictions_frozen_sha256": None,
        "predictions_frozen_row_count": None,
        "result_manifest_sha256": None,
        "sealed_outcome_sha256": None,
        "failure": None,
        "freeze_evidence": evidence,
    }
    _create_exclusive_json(claim_file, claim)
    try:
        running_hash = _replace_gate_status(
            gate_path=gate_file,
            expected_status="RELEASED_FOR_SINGLE_EVALUATION",
            new_status="RUNNING_EXTERNAL_EVALUATION",
            expected_sha256=gate_hash_before,
        )
    except Exception as exc:
        claim["status"] = "FAILED_CLOSED"
        claim["failure"] = {
            "phase": "CLAIMING",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        claim["updated_at_utc"] = _utc_now()
        _atomic_write_json(claim_file, claim)
        raise
    claim["status"] = "RUNNING"
    claim["gate_sha256_running"] = running_hash
    claim["updated_at_utc"] = _utc_now()
    _atomic_write_json(claim_file, claim)
    return claim


def mark_predictions_frozen(
    *,
    gate_path: str | Path,
    claim_path: str | Path,
    run_id: str,
    predictions_path: str | Path,
    expected_rows: int,
) -> dict[str, Any]:
    gate = load_external_release_gate(gate_path)
    if gate.status != "RUNNING_EXTERNAL_EVALUATION":
        raise ExternalEvaluationTransactionError("只有 RUNNING 门可冻结预测")
    claim_file = Path(claim_path).resolve()
    claim = _read_claim(claim_file, run_id)
    if claim.get("status") != "RUNNING" or claim.get("outcome_access_started"):
        raise ExternalEvaluationTransactionError("claim 不在可冻结预测的阶段")
    prediction_file = Path(predictions_path).resolve()
    if not prediction_file.is_file():
        raise ExternalEvaluationTransactionError("冻结预测文件不存在")
    claim["predictions_frozen_sha256"] = sha256_file(prediction_file)
    claim["predictions_frozen_row_count"] = int(expected_rows)
    claim["phase"] = "PREDICTIONS_FROZEN"
    claim["updated_at_utc"] = _utc_now()
    _atomic_write_json(claim_file, claim)
    return claim


def mark_outcome_access_started(
    *, gate_path: str | Path, claim_path: str | Path, run_id: str
) -> dict[str, Any]:
    gate = load_external_release_gate(gate_path)
    if gate.status != "RUNNING_EXTERNAL_EVALUATION":
        raise ExternalEvaluationTransactionError("封存真值只能由 RUNNING 事务读取")
    claim_file = Path(claim_path).resolve()
    claim = _read_claim(claim_file, run_id)
    if claim.get("status") != "RUNNING":
        raise ExternalEvaluationTransactionError("claim 不在 RUNNING 状态")
    if claim.get("phase") != "PREDICTIONS_FROZEN":
        raise ExternalEvaluationTransactionError("必须先冻结全部预测再读真值")
    if not claim.get("predictions_frozen_sha256"):
        raise ExternalEvaluationTransactionError("缺少冻结预测 SHA-256")
    claim["outcome_access_started"] = True
    claim["phase"] = "OUTCOME_ACCESS_STARTED"
    claim["outcome_access_started_at_utc"] = _utc_now()
    claim["updated_at_utc"] = _utc_now()
    _atomic_write_json(claim_file, claim)
    return claim


def assert_sealed_access_authorized(
    *, gate_path: str | Path, claim_path: str | Path, run_id: str
) -> None:
    gate = load_external_release_gate(gate_path)
    claim = _read_claim(Path(claim_path).resolve(), run_id)
    if gate.status != "RUNNING_EXTERNAL_EVALUATION":
        raise ExternalHoldoutLocked("释放门不在 RUNNING 状态")
    if claim.get("status") != "RUNNING" or not claim.get("outcome_access_started"):
        raise ExternalHoldoutLocked("事务尚未授权读取封存真值")


def complete_external_evaluation(
    *,
    gate_path: str | Path,
    claim_path: str | Path,
    run_id: str,
    result_manifest_path: str | Path,
    sealed_outcome_sha256: str,
) -> dict[str, Any]:
    claim_file = Path(claim_path).resolve()
    claim = _read_claim(claim_file, run_id)
    if claim.get("status") != "RUNNING" or not claim.get("outcome_access_started"):
        raise ExternalEvaluationTransactionError("未读取真值的事务不能标记完成")
    manifest = Path(result_manifest_path).resolve()
    if not manifest.is_file():
        raise ExternalEvaluationTransactionError("最终运行清单不存在")
    gate_file = Path(gate_path).resolve()
    running_hash = str(claim.get("gate_sha256_running") or "")
    consumed_hash = _replace_gate_status(
        gate_path=gate_file,
        expected_status="RUNNING_EXTERNAL_EVALUATION",
        new_status="CONSUMED_EXTERNAL_EVALUATION",
        expected_sha256=running_hash,
    )
    claim["status"] = "CONSUMED"
    claim["phase"] = "COMPLETED"
    claim["result_manifest_sha256"] = sha256_file(manifest)
    claim["sealed_outcome_sha256"] = sealed_outcome_sha256
    claim["gate_sha256_consumed"] = consumed_hash
    claim["completed_at_utc"] = _utc_now()
    claim["updated_at_utc"] = _utc_now()
    _atomic_write_json(claim_file, claim)
    return claim


def fail_external_evaluation_closed(
    *,
    gate_path: str | Path,
    claim_path: str | Path,
    run_id: str,
    exception: BaseException,
) -> dict[str, Any]:
    """任何已 claim 的失败都不自动回退为 RELEASED。"""

    claim_file = Path(claim_path).resolve()
    claim = _read_claim(claim_file, run_id)
    gate_file = Path(gate_path).resolve()
    gate = load_external_release_gate(gate_file)
    if gate.status == "RUNNING_EXTERNAL_EVALUATION":
        failed_hash = _replace_gate_status(
            gate_path=gate_file,
            expected_status="RUNNING_EXTERNAL_EVALUATION",
            new_status="FAILED_CLOSED_EXTERNAL_EVALUATION",
            expected_sha256=str(claim.get("gate_sha256_running") or "") or None,
        )
    elif gate.status == "FAILED_CLOSED_EXTERNAL_EVALUATION":
        failed_hash = sha256_file(gate_file)
    else:
        raise ExternalEvaluationTransactionError(
            f"不能把当前门状态改为 FAILED_CLOSED: {gate.status}"
        )
    claim["status"] = "FAILED_CLOSED"
    claim["phase"] = (
        "FAILED_AFTER_OUTCOME_ACCESS"
        if claim.get("outcome_access_started")
        else "FAILED_BEFORE_OUTCOME_ACCESS"
    )
    claim["failure"] = {
        "phase": claim["phase"],
        "exception_type": type(exception).__name__,
        "message": str(exception),
    }
    claim["gate_sha256_failed_closed"] = failed_hash
    claim["failed_at_utc"] = _utc_now()
    claim["updated_at_utc"] = _utc_now()
    _atomic_write_json(claim_file, claim)
    return claim
