"""P3 开发期离线工作流对照实验。

本模块只读取 2024—2025 开发期工件。两个可运行实验臂共享冻结的
``PersistencePredictor`` 与同一组时间折外验证事件：

* ``deterministic_fixed_workflow``：直接调用本地冻结预测器；
* ``multi_agent_graph_offline``：执行 A1→A2→A4→A5 离线图。

预测完成并通过逐样本严格相等检查后，才读取隔离的开发期真值账本计算
MAE、RMSE 与 R²。模块不包含网络或 LLM 调用。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from copper_mas.agents.runtime import fixed_offline_plan, run_offline_graph
from copper_mas.contracts.cards import (
    AsOfAdmissionCardV2,
    ForecastRequestV2,
    ObservationCardV2,
    ObservationItemV2,
    PredictionCardV2,
    ProcessModeCardV2,
)
from copper_mas.contracts.runtime import AgentExecutionRecordV1, ResourceLedgerV1
from copper_mas.models.predictors import load_selected_predictor

RUN_ID = "P3_DEVELOPMENT_BENCHMARK_V1_2024_2025"
DIRECT_ARM = "deterministic_fixed_workflow"
GRAPH_ARM = "multi_agent_graph_offline"
SINGLE_LLM_ARM = "single_llm_agent"
EXPECTED_OOF_EVENTS = 2732
EXPECTED_FOLDS = 5
DEVELOPMENT_YEARS = frozenset({2024, 2025})
TARGETS = (
    ("target_cu_g_l", "predicted_cu_g_l", "Cu", "g/L"),
    ("target_as_mg_l", "predicted_as_mg_l", "As", "mg/L"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    return value


def _as_bool(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "是": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "否": False,
        "": False,
    }
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"{column} 含无法解析的布尔值: {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少必要字段: {missing}")


def _parse_time(values: pd.Series, *, source: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"{source} 含无法解析的时间")
    unexpected = sorted(set(parsed.dt.year.astype(int)) - DEVELOPMENT_YEARS)
    if unexpected:
        raise ValueError(f"P3 开发期只允许 2024—2025；{source} 检出 {unexpected}")
    return parsed


def _parse_json_list(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("预期 JSON 数组")
    return tuple(str(item) for item in parsed)


def _parse_json_counts(value: Any) -> dict[str, int]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("预期 JSON 对象")
    return {str(key): int(item) for key, item in parsed.items()}


def _validate_input_path(path: Path) -> None:
    normalized_parts = {part.strip().lower() for part in Path(path).resolve().parts}
    if "external_2026_locked" in normalized_parts:
        raise ValueError(f"P3 开发期基准禁止读取 2026 工件: {path}")


def reconstruct_p2_oof_identity(
    training_index: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    p2_manifest: dict[str, Any],
    *,
    expected_events: int = EXPECTED_OOF_EVENTS,
    expected_folds: int = EXPECTED_FOLDS,
) -> pd.DataFrame:
    """由冻结折角色重建 P2 的唯一 OOF 验证事件，不读取逐样本真值预测表。"""

    _require_columns(
        training_index,
        {
            "pair_id",
            "origin_event_id",
            "origin_recorded_at",
            "tier1_core_primary_prospective",
        },
        source="training_evaluation_index_v2.csv",
    )
    _require_columns(
        fold_manifest,
        {
            "fold_id",
            "fold_role",
            "pair_id",
            "origin_event_id",
            "decision_at",
            "preprocessing_fit_scope",
        },
        source="cv_fold_manifest_v2.csv",
    )
    if training_index["pair_id"].duplicated().any():
        raise ValueError("training_evaluation_index 的 pair_id 不唯一")
    if training_index["origin_event_id"].duplicated().any():
        raise ValueError("training_evaluation_index 的 origin_event_id 不唯一")
    if set(fold_manifest["preprocessing_fit_scope"].astype(str)) != {"TRAIN_FOLD_ONLY"}:
        raise ValueError("冻结折的预处理范围必须全部为 TRAIN_FOLD_ONLY")

    eligible = training_index.loc[
        _as_bool(
            training_index["tier1_core_primary_prospective"],
            column="tier1_core_primary_prospective",
        ),
        ["pair_id", "origin_event_id", "origin_recorded_at"],
    ].copy()
    validation = fold_manifest.loc[
        fold_manifest["fold_role"].astype(str).str.upper().eq("VALIDATION"),
        ["fold_id", "pair_id", "origin_event_id", "decision_at"],
    ].copy()
    if validation.duplicated(["fold_id", "pair_id"]).any():
        raise ValueError("同一验证折中出现重复 pair_id")
    cross_fold_counts = validation.groupby("pair_id")["fold_id"].nunique()
    if (cross_fold_counts > 1).any():
        examples = cross_fold_counts.loc[cross_fold_counts > 1].index[:5].tolist()
        raise ValueError(f"验证事件跨折重复: {examples}")

    sample = validation.merge(
        eligible,
        on=["pair_id", "origin_event_id"],
        how="inner",
        validate="one_to_one",
    )
    sample["decision_at"] = _parse_time(sample["decision_at"], source="cv_fold_manifest.decision_at")
    sample["origin_recorded_at"] = _parse_time(sample["origin_recorded_at"], source="training_index.origin_recorded_at")
    if not sample["decision_at"].eq(sample["origin_recorded_at"]).all():
        raise ValueError("OOF 决策时间与训练索引起点时间不一致")
    if sample["pair_id"].duplicated().any() or sample["origin_event_id"].duplicated().any():
        raise ValueError("重建后的 OOF 事件不唯一")

    expected_from_p2 = int((p2_manifest.get("counts") or {}).get("unique_oof_validation_pairs", -1))
    if expected_from_p2 != expected_events:
        raise ValueError(f"P2 清单记录的 OOF 数不是冻结值 {expected_events}: {expected_from_p2}")
    if len(sample) != expected_events:
        raise ValueError(f"重建 OOF 事件数应为 {expected_events}，实际 {len(sample)}")

    fold_counts = {str(key): int(value) for key, value in sample.groupby("fold_id", sort=True).size().items()}
    expected_fold_counts = {
        str(key): int(value)
        for key, value in ((p2_manifest.get("counts") or {}).get("fold_validation_counts") or {}).items()
    }
    if fold_counts != expected_fold_counts:
        raise ValueError(f"重建的逐折 OOF 数与 P2 清单不一致: {fold_counts} != {expected_fold_counts}")
    if len(fold_counts) != expected_folds:
        raise ValueError(f"OOF 必须覆盖 {expected_folds} 个冻结时间折")
    return sample.sort_values(["decision_at", "pair_id"], kind="mergesort").reset_index(drop=True)


def _make_admission(raw: pd.Series) -> AsOfAdmissionCardV2:
    return AsOfAdmissionCardV2(
        origin_event_id=str(raw["origin_event_id"]),
        decision_at=pd.Timestamp(raw["decision_at"]).to_pydatetime(),
        feature_cutoff_at=pd.Timestamp(raw["feature_cutoff_at"]).to_pydatetime(),
        admission_status=str(raw["admission_status"]),
        reason_codes=_parse_json_list(raw["reason_codes"]),
        feature_group_counts=_parse_json_counts(raw["feature_group_counts"]),
        missing_feature_groups=_parse_json_list(raw["missing_feature_groups"]),
        source_quality_warnings=_parse_json_list(raw["source_quality_warnings"]),
    )


def _make_request(*, admission: AsOfAdmissionCardV2, feature_row: dict[str, Any]) -> ForecastRequestV2:
    decision_at = admission.decision_at
    observations = ObservationCardV2(
        origin_event_id=admission.origin_event_id,
        decision_at=decision_at,
        observations=(
            ObservationItemV2(
                canonical_tag="origin_cu_g_l",
                value=float(feature_row["origin_cu_g_l"]),
                unit="g/L",
                event_at=decision_at,
                available_at=decision_at,
                matched_anchor_at=decision_at,
                age_hours=0,
                missing=False,
                quality_code="KNOWN_CURRENT_RESULT",
            ),
            ObservationItemV2(
                canonical_tag="origin_as_mg_l",
                value=float(feature_row["origin_as_mg_l"]),
                unit="mg/L",
                event_at=decision_at,
                available_at=decision_at,
                matched_anchor_at=decision_at,
                age_hours=0,
                missing=False,
                quality_code="KNOWN_CURRENT_RESULT",
            ),
        ),
    )
    return ForecastRequestV2(
        request_id=f"p3-dev::{admission.origin_event_id}",
        admission=admission,
        observations=observations,
    )


def _direct_resource_record(
    *, run_id: str, started_at: datetime, ended_at: datetime, latency_ms: float
) -> ResourceLedgerV1:
    record = AgentExecutionRecordV1(
        run_id=run_id,
        agent_id="A4",
        engine="LOCAL_NUMERIC_MODEL",
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=latency_ms,
        status="PASSED",
        reason_codes=("DIRECT_FROZEN_PERSISTENCE",),
    )
    return ResourceLedgerV1.from_records(run_id, (record,))


def _card_row(arm: str, pair_id: str, fold_id: str, card: Any) -> dict[str, Any]:
    payload = card.model_dump(mode="json")
    row: dict[str, Any] = {"arm": arm, "pair_id": pair_id, "fold_id": fold_id}
    for key, value in payload.items():
        row[key] = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else value
        )
    return row


def _resource_rows(
    *, arm: str, pair_id: str, fold_id: str, ledger: ResourceLedgerV1
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ledger_row = {
        "arm": arm,
        "pair_id": pair_id,
        "fold_id": fold_id,
        "run_id": ledger.run_id,
        "total_latency_ms": ledger.total_latency_ms,
        "total_calls": ledger.total_calls,
        "total_input_tokens": ledger.total_input_tokens,
        "total_output_tokens": ledger.total_output_tokens,
        "total_estimated_cost_cny": ledger.total_estimated_cost_cny,
    }
    records: list[dict[str, Any]] = []
    for item in ledger.records:
        payload = item.model_dump(mode="json")
        payload["arm"] = arm
        payload["pair_id"] = pair_id
        payload["fold_id"] = fold_id
        payload["reason_codes"] = json.dumps(payload["reason_codes"], ensure_ascii=False, separators=(",", ":"))
        records.append(payload)
    return ledger_row, records


def _metric_row(
    *,
    arm: str,
    target_name: str,
    display_name: str,
    unit: str,
    scope: str,
    fold_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "target_name": target_name,
        "target_display_name": display_name,
        "unit": unit,
        "scope": scope,
        "fold_id": fold_id,
        "sample_count": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _percentile(values: pd.Series, percentile: float) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.percentile(array, percentile)) if len(array) else float("nan")


def _render_report(
    *,
    sample: pd.DataFrame,
    overall: pd.DataFrame,
    operational: pd.DataFrame,
    modes: pd.DataFrame,
    parity: dict[str, Any],
    source_hash_verified: bool,
) -> str:
    lines = [
        "# P3 2024—2025 开发期离线工作流基准报告 V1",
        "",
        "## 1. 结论",
        "",
        (
            f"本次在 {len(sample):,} 个唯一时间折外验证事件上完成了两个可运行实验臂。"
            "固定工作流直接调用冻结 PersistencePredictor；离线多智能体图执行"
            " A1→A2→A4→A5，但 A4 使用同一预测器。"
        ),
        "",
        (
            f"逐样本数值一致性检查结果：Cu 与 As 均为严格相等，最大绝对差分别为 "
            f"{parity['max_abs_difference_cu']:.1f} 和 {parity['max_abs_difference_as']:.1f}。"
            "因此两臂的预测指标完全相同；多智能体图在本实验中的增量价值是工况卡、"
            "审计卡和节点资源轨迹，而不是数值精度提升。"
        ),
        "",
        "## 2. 数据与评价边界",
        "",
        "- 样本：`tier1_core_primary_prospective` 的冻结五折 OOF 验证并集。",
        f"- 唯一起点数：{len(sample):,}；任何 pair/origin 均只属于一个验证折。",
        "- 年份：仅 2024—2025 开发期；没有读取 2026 外部时序留出集。",
        "- 模型：冻结清单中的 `PERSISTENCE_CURRENT_RESULT_V1`。",
        "- 标签隔离：两个实验臂完成预测和逐样本相等检查后，才读取开发期 outcome ledger 计算指标。",
        f"- P2 输入哈希复核：{'通过' if source_hash_verified else '未通过'}。",
        "",
        "## 3. 预测指标",
        "",
        "| 实验臂 | 目标 | n | MAE | RMSE | R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in overall.sort_values(["target_name", "arm"], kind="mergesort").iterrows():
        digits = 4 if row["target_display_name"] == "Cu" else 3
        lines.append(
            f"| {row['arm']} | {row['target_display_name']} ({row['unit']}) | "
            f"{int(row['sample_count'])} | {row['mae']:.{digits}f} | "
            f"{row['rmse']:.{digits}f} | {row['r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 稳定性、延迟与资源",
            "",
            (
                "| 实验臂 | 尝试数 | 成功数 | 失败率 | 拒绝率 | 警告率 | p50 延迟(m"
                "s) | p95 延迟(ms) | 调用数 | 输入/输出 token | 成本(CNY) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in operational.sort_values("arm", kind="mergesort").iterrows():
        lines.append(
            f"| {row['arm']} | {int(row['attempted_count'])} | {int(row['success_count'])} | "
            f"{row['failure_rate']:.2%} | {row['rejection_rate']:.2%} | "
            f"{row['warning_rate']:.2%} | {row['p50_end_to_end_latency_ms']:.4f} | "
            f"{row['p95_end_to_end_latency_ms']:.4f} | {int(row['total_calls'])} | "
            f"{int(row['total_input_tokens'])}/{int(row['total_output_tokens'])} | "
            f"{row['total_estimated_cost_cny']:.6f} |"
        )
    lines.extend(
        [
            "",
            "延迟是本机单进程顺序运行的端到端观测值，仅用于建立当前实现基线；不能外推为生产部署 SLA。",
            "",
            "## 5. 工况识别",
            "",
            f"A2 共形成 {len(modes):,} 类工况组合。出现频率最高的工况如下：",
            "",
            "| 工况编码 | 事件数 | 比例 | 平均置信度 | 警告率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in modes.sort_values("event_count", ascending=False, kind="mergesort").head(10).iterrows():
        lines.append(
            f"| {row['mode_code']} | {int(row['event_count'])} | {row['event_rate']:.2%} | "
            f"{row['mean_confidence']:.4f} | {row['warning_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 6. Single-LLM 实验臂状态",
            "",
            (
                "`single_llm_agent` 标记为 `PENDING_LIVE_CALL_AUTHORIZATION`。本报"
                "告没有伪造该臂的精度、延迟、token 或成本；在密钥轮换、调用预算和现"
                "场数据出站边界确认前不运行。"
            ),
            "",
            "## 7. 结果解释边界",
            "",
            (
                "本基准证明离线多智能体图可以在不改变冻结数值预测的前提下增加结构化"
                "工况与审计轨迹。由于两个实验臂共用完全相同的 A4，预测指标相同是设"
                "计约束，不是多智能体提高精度的证据。后续若比较真实能力差异，应预先"
                "冻结不同的信息处理或路由机制，并继续使用相同 OOF/外部时序留出协议"
                "。"
            ),
            "",
            "所有逐事件卡片、资源账本、指标、状态、报告和 SHA-256 清单均保存在本目录。",
            "",
        ]
    )
    return "\n".join(lines)


def run_development_workflow_benchmark(
    *,
    input_dir: Path,
    p2_dir: Path,
    selected_model_manifest: Path,
    output_dir: Path,
    expected_events: int = EXPECTED_OOF_EVENTS,
) -> dict[str, Any]:
    """运行 P3 开发期离线对照并写出审计工件。"""

    run_started = time.perf_counter()
    generated_at = datetime.now(UTC)
    input_dir = Path(input_dir).resolve()
    p2_dir = Path(p2_dir).resolve()
    selected_model_manifest = Path(selected_model_manifest).resolve()
    output_dir = Path(output_dir).resolve()

    sources = {
        "features": input_dir / "core_feature_matrix_v2.csv",
        "admission": input_dir / "as_of_admission_card_core_v2.csv",
        "index": input_dir / "training_evaluation_index_v2.csv",
        "folds": input_dir / "cv_fold_manifest_v2.csv",
        "outcomes": input_dir / "outcome_ledger_v2.csv",
        "p2_manifest": p2_dir / "run_manifest_v1.json",
        "selected_model": selected_model_manifest,
    }
    for path in sources.values():
        _validate_input_path(path)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P3 开发期基准输入缺失: {missing}")

    selected_payload = json.loads(sources["selected_model"].read_text(encoding="utf-8"))
    if selected_payload.get("external_2026_read") is not False:
        raise ValueError("冻结模型清单必须明确 external_2026_read=false")
    if set(selected_payload.get("development_years") or []) != DEVELOPMENT_YEARS:
        raise ValueError("冻结模型清单的开发年份必须是 2024 与 2025")
    predictor = load_selected_predictor(sources["selected_model"])

    p2_manifest = json.loads(sources["p2_manifest"].read_text(encoding="utf-8"))
    safety = p2_manifest.get("safety") or {}
    if safety.get("external_2026_read") is not False:
        raise ValueError("P2 清单未证明 external_2026_read=false")
    if p2_manifest.get("sample") != "tier1_core_primary_prospective":
        raise ValueError("P2 样本口径不是 tier1_core_primary_prospective")

    # 此阶段不读取 outcome ledger；先完成输入身份复核、预测和数值一致性审计。
    features = pd.read_csv(sources["features"], low_memory=False)
    admissions = pd.read_csv(sources["admission"], low_memory=False)
    training_index = pd.read_csv(sources["index"], low_memory=False)
    folds = pd.read_csv(sources["folds"], low_memory=False)
    _require_columns(
        features,
        {"origin_event_id", "decision_at", "origin_cu_g_l", "origin_as_mg_l"},
        source="core_feature_matrix_v2.csv",
    )
    _require_columns(
        admissions,
        {
            "origin_event_id",
            "decision_at",
            "feature_cutoff_at",
            "admission_status",
            "reason_codes",
            "feature_group_counts",
            "missing_feature_groups",
            "source_quality_warnings",
        },
        source="as_of_admission_card_core_v2.csv",
    )
    if features["origin_event_id"].duplicated().any():
        raise ValueError("核心特征矩阵 origin_event_id 不唯一")
    if admissions["origin_event_id"].duplicated().any():
        raise ValueError("准入卡 origin_event_id 不唯一")
    features["decision_at"] = _parse_time(features["decision_at"], source="core_feature_matrix.decision_at")
    admissions["decision_at"] = _parse_time(admissions["decision_at"], source="as_of_admission.decision_at")
    admissions["feature_cutoff_at"] = _parse_time(
        admissions["feature_cutoff_at"], source="as_of_admission.feature_cutoff_at"
    )

    p2_expected_hashes = p2_manifest.get("source_sha256") or {}
    pre_prediction_hashes = {
        "core_feature_matrix_v2.csv": sha256_file(sources["features"]),
        "training_evaluation_index_v2.csv": sha256_file(sources["index"]),
        "cv_fold_manifest_v2.csv": sha256_file(sources["folds"]),
    }
    for filename, actual_hash in pre_prediction_hashes.items():
        if p2_expected_hashes.get(filename) != actual_hash:
            raise ValueError(f"{filename} 与 P2 运行时输入哈希不一致")

    sample = reconstruct_p2_oof_identity(training_index, folds, p2_manifest, expected_events=expected_events)
    feature_by_origin = features.set_index("origin_event_id", drop=False)
    admission_by_origin = admissions.set_index("origin_event_id", drop=False)
    missing_feature_ids = sorted(set(sample["origin_event_id"]) - set(feature_by_origin.index))
    missing_admission_ids = sorted(set(sample["origin_event_id"]) - set(admission_by_origin.index))
    if missing_feature_ids or missing_admission_ids:
        raise ValueError(f"OOF 输入不完整；特征缺失 {missing_feature_ids[:5]}，准入卡缺失 {missing_admission_ids[:5]}")

    event_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    for _, identity in sample.iterrows():
        pair_id = str(identity["pair_id"])
        origin_event_id = str(identity["origin_event_id"])
        fold_id = str(identity["fold_id"])
        feature_series = feature_by_origin.loc[origin_event_id]
        admission_series = admission_by_origin.loc[origin_event_id]
        if isinstance(feature_series, pd.DataFrame) or isinstance(admission_series, pd.DataFrame):
            raise ValueError(f"起点 {origin_event_id} 的特征或准入卡不唯一")
        feature_row = feature_series.to_dict()
        admission = _make_admission(admission_series)
        if admission.decision_at != pd.Timestamp(identity["decision_at"]).to_pydatetime():
            raise ValueError(f"起点 {origin_event_id} 的准入时间与 OOF 时间不一致")

        base_event = {
            "pair_id": pair_id,
            "origin_event_id": origin_event_id,
            "fold_id": fold_id,
            "decision_at": pd.Timestamp(identity["decision_at"]).isoformat(),
        }
        if admission.admission_status == "REJECTED":
            event_rows.append(
                {
                    **base_event,
                    "direct_status": "REJECTED",
                    "graph_status": "REJECTED",
                    "direct_predicted_cu_g_l": np.nan,
                    "graph_predicted_cu_g_l": np.nan,
                    "cu_abs_difference": np.nan,
                    "direct_predicted_as_mg_l": np.nan,
                    "graph_predicted_as_mg_l": np.nan,
                    "as_abs_difference": np.nan,
                    "numeric_exact_match": False,
                    "direct_end_to_end_latency_ms": 0.0,
                    "graph_end_to_end_latency_ms": 0.0,
                    "direct_warning_count": len(admission.reason_codes),
                    "graph_warning_count": len(admission.reason_codes),
                    "mode_code": "NOT_RUN_ADMISSION_REJECTED",
                    "mode_confidence": 0.0,
                    "audit_passed": False,
                }
            )
            continue

        request = _make_request(admission=admission, feature_row=feature_row)
        unused_mode = ProcessModeCardV2(
            origin_event_id=origin_event_id,
            decision_at=admission.decision_at,
            mode_code="NOT_USED_BY_PERSISTENCE",
            confidence=0,
            evidence_observation_ids=(),
            warnings=(),
        )

        direct_run_id = f"{RUN_ID}::DIRECT::{pair_id}"
        direct_started_at = datetime.now(UTC)
        direct_started_perf = time.perf_counter()
        direct_numeric = predictor(feature_row, unused_mode)
        direct_latency_ms = (time.perf_counter() - direct_started_perf) * 1000
        direct_ended_at = datetime.now(UTC)
        direct_resources = _direct_resource_record(
            run_id=direct_run_id,
            started_at=direct_started_at,
            ended_at=direct_ended_at,
            latency_ms=direct_latency_ms,
        )
        direct_prediction = PredictionCardV2(
            request_id=request.request_id,
            origin_event_id=origin_event_id,
            decision_at=admission.decision_at,
            frozen_model_id=predictor.model_id,
            **direct_numeric.model_dump(),
        )

        graph_run_id = f"{RUN_ID}::GRAPH::{pair_id}"
        graph_plan = fixed_offline_plan(
            run_id=graph_run_id,
            numeric_predictor_ref=predictor.model_id,
            post_freeze=False,
        )
        graph_started_perf = time.perf_counter()
        graph_result = run_offline_graph(
            plan=graph_plan,
            request=request,
            feature_row=feature_row,
            predict=predictor,
        )
        graph_latency_ms = (time.perf_counter() - graph_started_perf) * 1000

        cu_equal = direct_prediction.predicted_cu_g_l == graph_result.prediction.predicted_cu_g_l
        as_equal = direct_prediction.predicted_as_mg_l == graph_result.prediction.predicted_as_mg_l
        direct_warning_codes = list(admission.reason_codes) + list(admission.source_quality_warnings)
        graph_warning_codes: list[str] = []
        for record in graph_result.resources.records:
            if record.status == "WARNING":
                graph_warning_codes.extend(record.reason_codes)
                for code in record.reason_codes:
                    warning_rows.append(
                        {
                            **base_event,
                            "arm": GRAPH_ARM,
                            "agent_id": record.agent_id,
                            "warning_code": code,
                        }
                    )
        for code in direct_warning_codes:
            warning_rows.append(
                {
                    **base_event,
                    "arm": DIRECT_ARM,
                    "agent_id": "INPUT_ADMISSION",
                    "warning_code": code,
                }
            )

        event_rows.append(
            {
                **base_event,
                "direct_status": "PASSED",
                "graph_status": "PASSED" if graph_result.audit.passed else "FAILED",
                "direct_predicted_cu_g_l": direct_prediction.predicted_cu_g_l,
                "graph_predicted_cu_g_l": graph_result.prediction.predicted_cu_g_l,
                "cu_abs_difference": abs(direct_prediction.predicted_cu_g_l - graph_result.prediction.predicted_cu_g_l),
                "direct_predicted_as_mg_l": direct_prediction.predicted_as_mg_l,
                "graph_predicted_as_mg_l": graph_result.prediction.predicted_as_mg_l,
                "as_abs_difference": abs(
                    direct_prediction.predicted_as_mg_l - graph_result.prediction.predicted_as_mg_l
                ),
                "numeric_exact_match": bool(cu_equal and as_equal),
                "direct_end_to_end_latency_ms": direct_latency_ms,
                "graph_end_to_end_latency_ms": graph_latency_ms,
                "direct_warning_count": len(set(direct_warning_codes)),
                "graph_warning_count": len(set(graph_warning_codes)),
                "mode_code": graph_result.process_mode.mode_code,
                "mode_confidence": graph_result.process_mode.confidence,
                "audit_passed": graph_result.audit.passed,
            }
        )
        mode_rows.append(_card_row(GRAPH_ARM, pair_id, fold_id, graph_result.process_mode))
        prediction_rows.extend(
            [
                _card_row(DIRECT_ARM, pair_id, fold_id, direct_prediction),
                _card_row(GRAPH_ARM, pair_id, fold_id, graph_result.prediction),
            ]
        )
        audit_rows.append(_card_row(GRAPH_ARM, pair_id, fold_id, graph_result.audit))
        direct_ledger_row, direct_record_rows = _resource_rows(
            arm=DIRECT_ARM,
            pair_id=pair_id,
            fold_id=fold_id,
            ledger=direct_resources,
        )
        graph_ledger_row, graph_record_rows = _resource_rows(
            arm=GRAPH_ARM,
            pair_id=pair_id,
            fold_id=fold_id,
            ledger=graph_result.resources,
        )
        resource_rows.extend([direct_ledger_row, graph_ledger_row])
        execution_rows.extend([*direct_record_rows, *graph_record_rows])

    events = pd.DataFrame(event_rows)
    if len(events) != expected_events:
        raise RuntimeError(f"工作流运行事件数异常: {len(events)} != {expected_events}")
    if not events["numeric_exact_match"].all():
        mismatch = events.loc[~events["numeric_exact_match"], "pair_id"].head(5).tolist()
        raise RuntimeError(f"两个实验臂的逐样本数值预测不完全一致: {mismatch}")
    if not events["audit_passed"].all():
        failed = events.loc[~events["audit_passed"], "pair_id"].head(5).tolist()
        raise RuntimeError(f"多智能体图审计未全部通过: {failed}")

    parity: dict[str, Any] = {
        "status": "PASSED_EXACT_SAMPLEWISE_EQUALITY",
        "comparison_event_count": int(len(events)),
        "cu_exact_match_count": int((events["cu_abs_difference"].astype(float) == 0.0).sum()),
        "as_exact_match_count": int((events["as_abs_difference"].astype(float) == 0.0).sum()),
        "max_abs_difference_cu": float(events["cu_abs_difference"].max()),
        "max_abs_difference_as": float(events["as_abs_difference"].max()),
        "same_frozen_predictor_id": predictor.model_id,
        "external_2026_read": False,
        "llm_calls_made": 0,
    }

    # 只有预测与严格相等检查完成后，才读取隔离的开发期真值账本。
    outcomes = pd.read_csv(sources["outcomes"], low_memory=False)
    _require_columns(
        outcomes,
        {"pair_id", "target_cu_g_l", "target_as_mg_l"},
        source="outcome_ledger_v2.csv",
    )
    if outcomes["pair_id"].duplicated().any():
        raise ValueError("开发期真值账本 pair_id 不唯一")
    expected_outcome_hash = p2_expected_hashes.get("outcome_ledger_v2.csv")
    actual_outcome_hash = sha256_file(sources["outcomes"])
    if actual_outcome_hash != expected_outcome_hash:
        raise ValueError("outcome_ledger_v2.csv 与 P2 运行时输入哈希不一致")
    evaluated = events.merge(
        outcomes[["pair_id", "target_cu_g_l", "target_as_mg_l"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not evaluated["_merge"].eq("both").all():
        missing_pairs = evaluated.loc[evaluated["_merge"] != "both", "pair_id"].head(5).tolist()
        raise ValueError(f"OOF 样本缺少开发期真值: {missing_pairs}")
    evaluated = evaluated.drop(columns="_merge")
    if evaluated[["target_cu_g_l", "target_as_mg_l"]].isna().any().any():
        raise ValueError("OOF 开发期 Cu/As 真值不完整")

    overall_metric_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    prediction_columns = {
        DIRECT_ARM: {
            "predicted_cu_g_l": "direct_predicted_cu_g_l",
            "predicted_as_mg_l": "direct_predicted_as_mg_l",
        },
        GRAPH_ARM: {
            "predicted_cu_g_l": "graph_predicted_cu_g_l",
            "predicted_as_mg_l": "graph_predicted_as_mg_l",
        },
    }
    for target_name, prediction_name, display_name, unit in TARGETS:
        y_true_all = evaluated[target_name].to_numpy(dtype=float)
        for arm, mapping in prediction_columns.items():
            y_pred_all = evaluated[mapping[prediction_name]].to_numpy(dtype=float)
            overall_metric_rows.append(
                _metric_row(
                    arm=arm,
                    target_name=target_name,
                    display_name=display_name,
                    unit=unit,
                    scope="POOLED_OOF",
                    fold_id="ALL_FOLDS",
                    y_true=y_true_all,
                    y_pred=y_pred_all,
                )
            )
            for fold_id, fold_group in evaluated.groupby("fold_id", sort=True):
                fold_metric_rows.append(
                    _metric_row(
                        arm=arm,
                        target_name=target_name,
                        display_name=display_name,
                        unit=unit,
                        scope="SINGLE_FOLD",
                        fold_id=str(fold_id),
                        y_true=fold_group[target_name].to_numpy(dtype=float),
                        y_pred=fold_group[mapping[prediction_name]].to_numpy(dtype=float),
                    )
                )
    overall_metrics = pd.DataFrame(overall_metric_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    for target_name, *_ in TARGETS:
        subset = overall_metrics.loc[overall_metrics["target_name"] == target_name].sort_values("arm")
        numeric_columns = ["mae", "rmse", "r2"]
        if len(subset) != 2 or any(subset[column].nunique(dropna=False) != 1 for column in numeric_columns):
            raise RuntimeError(f"{target_name} 的两臂汇总指标不完全一致")
    parity["overall_metrics_exact_match"] = True

    resources = pd.DataFrame(resource_rows)
    executions = pd.DataFrame(execution_rows)
    modes = pd.DataFrame(mode_rows)
    predictions = pd.DataFrame(prediction_rows)
    audits = pd.DataFrame(audit_rows)
    warnings = pd.DataFrame(warning_rows)

    operational_rows: list[dict[str, Any]] = []
    for arm in (DIRECT_ARM, GRAPH_ARM):
        status_column = "direct_status" if arm == DIRECT_ARM else "graph_status"
        latency_column = "direct_end_to_end_latency_ms" if arm == DIRECT_ARM else "graph_end_to_end_latency_ms"
        warning_column = "direct_warning_count" if arm == DIRECT_ARM else "graph_warning_count"
        attempted = len(events)
        success = int(events[status_column].eq("PASSED").sum())
        failed = int(events[status_column].eq("FAILED").sum())
        rejected = int(events[status_column].eq("REJECTED").sum())
        arm_resources = resources.loc[resources["arm"] == arm]
        operational_rows.append(
            {
                "arm": arm,
                "execution_status": "COMPLETED",
                "attempted_count": attempted,
                "success_count": success,
                "failure_count": failed,
                "rejected_count": rejected,
                "warning_event_count": int((events[warning_column] > 0).sum()),
                "failure_rate": failed / attempted,
                "rejection_rate": rejected / attempted,
                "warning_rate": float((events[warning_column] > 0).mean()),
                "p50_end_to_end_latency_ms": _percentile(events[latency_column], 50),
                "p95_end_to_end_latency_ms": _percentile(events[latency_column], 95),
                "total_calls": int(arm_resources["total_calls"].sum()),
                "total_input_tokens": int(arm_resources["total_input_tokens"].sum()),
                "total_output_tokens": int(arm_resources["total_output_tokens"].sum()),
                "total_estimated_cost_cny": float(arm_resources["total_estimated_cost_cny"].sum()),
            }
        )
    operational = pd.DataFrame(operational_rows)

    agent_summary_rows: list[dict[str, Any]] = []
    for (arm, agent_id), group in executions.groupby(["arm", "agent_id"], sort=True):
        agent_summary_rows.append(
            {
                "arm": arm,
                "agent_id": agent_id,
                "execution_count": int(len(group)),
                "warning_count": int(group["status"].eq("WARNING").sum()),
                "failure_count": int(group["status"].eq("FAILED").sum()),
                "p50_latency_ms": _percentile(group["latency_ms"], 50),
                "p95_latency_ms": _percentile(group["latency_ms"], 95),
                "total_calls": int(group["call_count"].sum()),
                "total_input_tokens": int(group["input_tokens"].sum()),
                "total_output_tokens": int(group["output_tokens"].sum()),
                "total_estimated_cost_cny": float(group["estimated_cost_cny"].sum()),
            }
        )
    agent_summary = pd.DataFrame(agent_summary_rows)

    mode_events = events.loc[events["graph_status"] == "PASSED"].copy()
    mode_distribution = (
        mode_events.groupby("mode_code", sort=True)
        .agg(
            event_count=("pair_id", "size"),
            mean_confidence=("mode_confidence", "mean"),
            warning_event_count=("graph_warning_count", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    mode_distribution["event_rate"] = mode_distribution["event_count"] / len(mode_events)
    mode_distribution["warning_rate"] = mode_distribution["warning_event_count"] / mode_distribution["event_count"]
    warning_distribution = (
        warnings.groupby(["arm", "agent_id", "warning_code"], sort=True).size().rename("event_count").reset_index()
        if not warnings.empty
        else pd.DataFrame(columns=["arm", "agent_id", "warning_code", "event_count"])
    )
    fold_coverage = (
        sample.groupby("fold_id", sort=True)
        .agg(
            oof_event_count=("pair_id", "size"),
            unique_pair_count=("pair_id", "nunique"),
            unique_origin_count=("origin_event_id", "nunique"),
            decision_at_min=("decision_at", "min"),
            decision_at_max=("decision_at", "max"),
        )
        .reset_index()
    )

    arm_registry = pd.DataFrame(
        [
            {
                "arm": DIRECT_ARM,
                "execution_status": "COMPLETED",
                "results_available": True,
                "external_llm_calls_made": 0,
                "reason": "冻结 PersistencePredictor 直接本地运行",
            },
            {
                "arm": GRAPH_ARM,
                "execution_status": "COMPLETED",
                "results_available": True,
                "external_llm_calls_made": 0,
                "reason": "A1→A2→A4→A5 全部离线确定性运行",
            },
            {
                "arm": SINGLE_LLM_ARM,
                "execution_status": "PENDING_LIVE_CALL_AUTHORIZATION",
                "results_available": False,
                "external_llm_calls_made": np.nan,
                "reason": "等待密钥轮换、调用预算与数据出站授权；未伪造指标",
            },
        ]
    )
    single_llm_status = {
        "arm": SINGLE_LLM_ARM,
        "status": "PENDING_LIVE_CALL_AUTHORIZATION",
        "results_available": False,
        "metrics": None,
        "latency": None,
        "tokens": None,
        "cost": None,
        "live_calls_made": 0,
        "reason_codes": [
            "ROTATED_KEYS_NOT_CONFIGURED",
            "MONTHLY_BUDGET_NOT_CONFIRMED",
            "DATA_EGRESS_BOUNDARY_NOT_AUTHORIZED",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_artifacts = {
        "event_prediction_comparison_v1.csv": events,
        "process_mode_ledger_v2.csv": modes,
        "prediction_ledger_v2.csv": predictions,
        "audit_ledger_v2.csv": audits,
        "resource_ledger_v1.csv": resources,
        "agent_execution_ledger_v1.csv": executions,
        "overall_predictive_metrics_v1.csv": overall_metrics,
        "fold_predictive_metrics_v1.csv": fold_metrics,
        "workflow_operational_metrics_v1.csv": operational,
        "agent_resource_summary_v1.csv": agent_summary,
        "mode_distribution_v1.csv": mode_distribution,
        "warning_code_distribution_v1.csv": warning_distribution,
        "oof_fold_coverage_v1.csv": fold_coverage,
        "experiment_arm_registry_v1.csv": arm_registry,
    }
    for filename, frame in csv_artifacts.items():
        frame.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
            float_format="%.10g",
        )
    _write_json(output_dir / "numeric_parity_audit_v1.json", parity)
    _write_json(output_dir / "single_llm_agent_status_v1.json", single_llm_status)

    source_hashes = {path.name: sha256_file(path) for path in sources.values()}
    source_hash_verified = all(
        source_hashes.get(filename) == expected_hash
        for filename, expected_hash in p2_expected_hashes.items()
        if filename in source_hashes
    )
    report = _render_report(
        sample=sample,
        overall=overall_metrics,
        operational=operational,
        modes=mode_distribution,
        parity=parity,
        source_hash_verified=source_hash_verified,
    )
    report_path = output_dir / "P3开发期离线工作流基准报告_V1.md"
    report_path.write_text(report, encoding="utf-8")

    primary_paths = [
        *(output_dir / filename for filename in csv_artifacts),
        output_dir / "numeric_parity_audit_v1.json",
        output_dir / "single_llm_agent_status_v1.json",
        report_path,
    ]
    hash_rows = [
        {
            "artifact": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in primary_paths
    ]
    hash_frame = pd.DataFrame(hash_rows).sort_values("artifact", kind="mergesort")
    hash_path = output_dir / "artifact_hashes_v1.csv"
    hash_frame.to_csv(hash_path, index=False, encoding="utf-8-sig")

    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "generated_at_utc": generated_at.isoformat(),
        "duration_seconds": time.perf_counter() - run_started,
        "development_years": [2024, 2025],
        "sample": "tier1_core_primary_prospective_unique_oof_validation_union",
        "counts": {
            "unique_oof_events": int(len(sample)),
            "fold_count": int(sample["fold_id"].nunique()),
            "fold_validation_counts": {
                str(key): int(value) for key, value in sample.groupby("fold_id", sort=True).size().items()
            },
            "completed_arms": 2,
            "pending_arms": 1,
            "mode_count": int(len(mode_distribution)),
        },
        "arms": {
            DIRECT_ARM: "COMPLETED",
            GRAPH_ARM: "COMPLETED",
            SINGLE_LLM_ARM: "PENDING_LIVE_CALL_AUTHORIZATION",
        },
        "graph_order": ["A1", "A2", "A4", "A5"],
        "numeric_predictor_ref": predictor.model_id,
        "numeric_parity": parity,
        "safety": {
            "external_2026_read": False,
            "llm_calls_made": 0,
            "network_calls_made": 0,
            "api_keys_read": False,
            "outcome_ledger_read_phase": "AFTER_PREDICTION_AND_NUMERIC_PARITY",
            "p2_oof_prediction_file_read": False,
            "oof_identity_reconstructed_from_frozen_fold_roles": True,
            "random_split_used": False,
            "preprocessing_fit_scope": "TRAIN_FOLD_ONLY",
        },
        "source_sha256": source_hashes,
        "p2_source_hashes_verified": source_hash_verified,
        "output_sha256": {
            **{row["artifact"]: row["sha256"] for row in hash_rows},
            hash_path.name: sha256_file(hash_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation": (
            "两臂共用同一冻结 A4；相同预测指标是设计约束。多智能体图的本轮增量是工况、审计与资源轨迹，不代表精度提升。"
        ),
    }
    _write_json(output_dir / "run_manifest_v1.json", manifest)
    return manifest
