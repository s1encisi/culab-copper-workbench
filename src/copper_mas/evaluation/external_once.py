"""一次性 2026 外部时间留出评价器。

正式运行顺序被强制为：校验冻结工件 -> 原子 claim -> 生成并冻结 682 条
预测 -> 记录 outcome access started -> 读取封存真值 -> 计算预注册指标 ->
原子发布 -> 消费释放门。模块不包含训练、调参、网络或 LLM 调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
import yaml

from copper_mas.agents.runtime import fixed_offline_plan, run_offline_graph
from copper_mas.contracts.cards import (
    AsOfAdmissionCardV2,
    ForecastRequestV2,
    ObservationCardV2,
    ObservationItemV2,
)
from copper_mas.evaluation.holdout import load_external_release_gate
from copper_mas.evaluation.transaction import (
    ExternalEvaluationTransactionError,
    assert_sealed_access_authorized,
    claim_external_evaluation,
    complete_external_evaluation,
    fail_external_evaluation_closed,
    mark_outcome_access_started,
    mark_predictions_frozen,
    sha256_file,
    verify_runtime_freeze_manifest,
)
from copper_mas.models.predictors import load_selected_predictor


EXPECTED_OUTCOME_COLUMNS = (
    "schema_version",
    "contract_id",
    "pair_id",
    "target_cu_g_l",
    "target_as_mg_l",
)
EXPECTED_ADMISSION_COLUMNS = {
    "schema_version",
    "contract_id",
    "origin_event_id",
    "decision_at",
    "feature_cutoff_at",
    "admission_status",
    "reason_codes",
    "feature_group_counts",
    "missing_feature_groups",
    "source_quality_warnings",
    "core_process_4_of_7_ready",
}
EXPECTED_ELIGIBILITY_COLUMNS = {
    "schema_version",
    "contract_id",
    "pair_id",
    "origin_event_id",
    "target_event_id",
    "decision_at",
    "target_recorded_at",
    "target_sample_at_assumed",
    "lead_to_recorded_hours",
    "prospectively_sampled",
    "target_source_quality_eligible",
    "evaluation_eligible",
    "evaluation_reason_codes",
}
EXPECTED_INDEX_COLUMNS = {
    "pair_id",
    "origin_event_id",
    "origin_recorded_at",
    "pair_status",
    "admission_status",
    "core_process_4_of_7_ready",
    "prospectively_sampled",
    "evaluation_eligible",
    "sealed_label_available",
    "partition_role",
    "runtime_prediction_available",
    "tier1_core_external_evaluation",
}
FORBIDDEN_PREDICTION_COLUMNS = {
    "target_cu_g_l",
    "target_as_mg_l",
    "observed_cu_g_l",
    "observed_as_mg_l",
}
TARGETS = (
    ("target_cu_g_l", "predicted_cu_g_l", "Cu", "g/L"),
    ("target_as_mg_l", "predicted_as_mg_l", "As", "mg/L"),
)


class BootstrapProtocolV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["MOVING_BLOCK_BOOTSTRAP"]
    scope: Literal["OVERALL_ONLY"]
    block_length_events: int = Field(gt=1)
    replicates: int = Field(ge=100)
    random_seed: int
    lower_percentile: float = Field(ge=0, lt=50)
    upper_percentile: float = Field(gt=50, le=100)


class ExternalEvaluationProtocolV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_id: Literal["EXTERNAL_2026_SINGLE_EVALUATION_PROTOCOL_V1"]
    schema_version: Literal["1.0"]
    partition_role: Literal["EXTERNAL_TEMPORAL_HOLDOUT"]
    expected_event_count: int = Field(gt=1)
    expected_pair_count: int = Field(gt=0)
    external_build_manifest_path: str
    external_build_manifest_sha256: str
    primary_metrics: tuple[
        Literal["MAE", "RMSE", "R2", "MEDAE", "P90AE", "BIAS"], ...
    ]
    metric_population: Literal["PASSED_AND_AUDIT_PASSED_ONLY"]
    operational_population: Literal["ALL_682_PREREGISTERED_PAIRS"]
    outcome_missing_policy: Literal["FAIL_CLOSED"]
    fixed_strata: tuple[
        Literal["OVERALL", "CALENDAR_MONTH", "LEAD_TIME_BIN", "PROCESS_MODE", "COVERAGE"],
        ...,
    ]
    lead_time_bin_edges_hours: tuple[float, ...]
    lead_time_bin_labels: tuple[str, ...]
    minimum_metric_sample_count: int = Field(default=1, gt=0)
    bootstrap: BootstrapProtocolV1
    prediction_interval_status: Literal[
        "NOT_APPLICABLE_NO_FROZEN_INTERVAL_MODEL"
    ]
    interval_coverage: Literal["NA"]
    llm_calls_allowed: Literal[False]
    network_calls_allowed: Literal[False]
    post_outcome_model_changes_allowed: Literal[False]
    post_outcome_strata_changes_allowed: Literal[False]

    def validate_lead_bins(self) -> None:
        if tuple(sorted(self.lead_time_bin_edges_hours)) != self.lead_time_bin_edges_hours:
            raise ValueError("lead time 分层边界必须严格递增")
        if len(self.lead_time_bin_labels) != len(self.lead_time_bin_edges_hours) + 1:
            raise ValueError("lead time 分层标签数必须等于边界数+1")


@dataclass(frozen=True)
class ExternalEvaluationPaths:
    project_root: Path
    gate_path: Path
    claim_path: Path
    review_decision_path: Path
    protocol_path: Path
    external_dir: Path
    output_dir: Path


def load_external_evaluation_protocol(
    path: str | Path,
) -> ExternalEvaluationProtocolV1:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    protocol = ExternalEvaluationProtocolV1.model_validate(payload)
    protocol.validate_lead_bins()
    return protocol


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], *, source: str, exact: bool = False
) -> None:
    required_set = set(required)
    actual = set(frame.columns)
    missing = sorted(required_set - actual)
    if missing:
        raise ValueError(f"{source} 缺少必要字段: {missing}")
    if exact and actual != required_set:
        raise ValueError(
            f"{source} 字段集不等于冻结协议: "
            f"多余={sorted(actual - required_set)}"
        )


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


def _make_request(
    *, admission: AsOfAdmissionCardV2, feature_row: dict[str, Any], run_id: str
) -> ForecastRequestV2:
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
        request_id=f"{run_id}::{admission.origin_event_id}",
        admission=admission,
        observations=observations,
    )


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _verify_declared_artifact(
    *, external_dir: Path, manifest: dict[str, Any], relative: str
) -> None:
    metadata = (manifest.get("artifacts") or {}).get(relative)
    if not isinstance(metadata, dict) or not metadata.get("sha256"):
        raise ValueError(f"外部构建清单缺少工件: {relative}")
    path = external_dir / relative
    if sha256_file(path).lower() != str(metadata["sha256"]).lower():
        raise ValueError(f"外部工件 SHA-256 不匹配: {relative}")


def _load_non_outcome_inputs(
    *, paths: ExternalEvaluationPaths, protocol: ExternalEvaluationProtocolV1
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = paths.project_root / protocol.external_build_manifest_path
    if sha256_file(manifest_path).lower() != protocol.external_build_manifest_sha256.lower():
        raise ValueError("外部构建清单 SHA-256 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("partition_role") != "EXTERNAL_TEMPORAL_HOLDOUT":
        raise ValueError("外部构建清单不是时间留出分区")
    counts = manifest.get("counts") or {}
    if int(counts.get("target_events", -1)) != protocol.expected_event_count:
        raise ValueError("外部事件数与预注册协议不一致")
    if int(counts.get("strict_adjacent_pairs", -1)) != protocol.expected_pair_count:
        raise ValueError("外部相邻对数与预注册协议不一致")

    consumed = (
        "core_feature_matrix_v2.csv",
        "as_of_admission_card_core_v2.csv",
        "evaluation_eligibility_ledger_v2.csv",
        "external_evaluation_index_v2.csv",
    )
    for relative in consumed:
        _verify_declared_artifact(
            external_dir=paths.external_dir, manifest=manifest, relative=relative
        )

    core = pd.read_csv(paths.external_dir / consumed[0], low_memory=False)
    admissions = pd.read_csv(paths.external_dir / consumed[1], low_memory=False)
    eligibility = pd.read_csv(paths.external_dir / consumed[2], low_memory=False)
    external_index = pd.read_csv(paths.external_dir / consumed[3], low_memory=False)
    _require_columns(
        core,
        {"origin_event_id", "decision_at", "origin_cu_g_l", "origin_as_mg_l"},
        source=consumed[0],
    )
    _require_columns(admissions, EXPECTED_ADMISSION_COLUMNS, source=consumed[1])
    _require_columns(eligibility, EXPECTED_ELIGIBILITY_COLUMNS, source=consumed[2])
    _require_columns(external_index, EXPECTED_INDEX_COLUMNS, source=consumed[3])
    if FORBIDDEN_PREDICTION_COLUMNS & set(external_index.columns):
        raise ValueError("外部主索引出现真值列")

    for frame, name, key in (
        (core, consumed[0], "origin_event_id"),
        (admissions, consumed[1], "origin_event_id"),
        (eligibility, consumed[2], "pair_id"),
        (external_index, consumed[3], "pair_id"),
    ):
        if frame[key].duplicated().any():
            raise ValueError(f"{name} 主键不唯一: {key}")

    eligibility["evaluation_eligible"] = _as_bool(
        eligibility["evaluation_eligible"], column="evaluation_eligible"
    )
    external_index["core_process_4_of_7_ready"] = _as_bool(
        external_index["core_process_4_of_7_ready"],
        column="core_process_4_of_7_ready",
    )
    if len(eligibility) != protocol.expected_pair_count:
        raise ValueError("评价资格账本不是预注册的相邻对数")
    if not eligibility["evaluation_eligible"].all():
        raise ValueError("预注册的 682 相邻对中出现不合格样本")
    eligibility["decision_at"] = pd.to_datetime(
        eligibility["decision_at"], errors="raise", format="mixed"
    )
    eligibility["target_recorded_at"] = pd.to_datetime(
        eligibility["target_recorded_at"], errors="raise", format="mixed"
    )
    if not eligibility["decision_at"].dt.year.eq(2026).all():
        raise ValueError("外部评价出现非 2026 决策时间")
    if not (eligibility["target_recorded_at"] > eligibility["decision_at"]).all():
        raise ValueError("外部评价目标时间不晚于决策时间")

    sample = (
        eligibility.merge(
            external_index[
                [
                    "pair_id",
                    "origin_event_id",
                    "core_process_4_of_7_ready",
                    "partition_role",
                ]
            ],
            on=["pair_id", "origin_event_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            admissions.drop(columns=["core_process_4_of_7_ready"]),
            on="origin_event_id",
            how="left",
            validate="one_to_one",
            suffixes=("_eligibility", "_admission"),
        )
        .merge(
            core,
            on="origin_event_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_feature"),
        )
    )
    if len(sample) != protocol.expected_pair_count or sample.isna().all(axis=1).any():
        raise ValueError("外部评价输入合并后行数异常")
    if not sample["partition_role"].eq("EXTERNAL_TEMPORAL_HOLDOUT").all():
        raise ValueError("外部评价样本分区角色异常")
    return manifest, sample.sort_values(
        ["decision_at_eligibility", "pair_id"], kind="mergesort"
    ).reset_index(drop=True)


def _run_frozen_predictions(
    *,
    sample: pd.DataFrame,
    selected_manifest_path: Path,
    run_id: str,
) -> pd.DataFrame:
    predictor = load_selected_predictor(selected_manifest_path)
    rows: list[dict[str, Any]] = []
    identity_columns = {
        "pair_id",
        "target_event_id",
        "target_recorded_at",
        "target_sample_at_assumed",
        "lead_to_recorded_hours",
        "prospectively_sampled",
        "target_source_quality_eligible",
        "evaluation_eligible",
        "evaluation_reason_codes",
        "core_process_4_of_7_ready",
        "partition_role",
    }
    admission_columns = set(EXPECTED_ADMISSION_COLUMNS)
    for raw in sample.itertuples(index=False, name=None):
        series = pd.Series(raw, index=sample.columns)
        pair_id = str(series["pair_id"])
        started = time.perf_counter()
        base: dict[str, Any] = {
            "run_id": run_id,
            "pair_id": pair_id,
            "origin_event_id": str(series["origin_event_id"]),
            "decision_at": pd.Timestamp(series["decision_at_eligibility"]).isoformat(),
            "target_recorded_at": pd.Timestamp(series["target_recorded_at"]).isoformat(),
            "lead_to_recorded_hours": float(series["lead_to_recorded_hours"]),
            "coverage_ready_4_of_7": bool(series["core_process_4_of_7_ready"]),
            "status": "FAILED",
            "failure_code": None,
            "frozen_model_id": predictor.model_id,
            "predicted_cu_g_l": np.nan,
            "predicted_as_mg_l": np.nan,
            "mode_code": None,
            "mode_confidence": np.nan,
            "mode_warnings": "[]",
            "audit_passed": False,
            "audit_checks": "[]",
            "total_latency_ms": np.nan,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_estimated_cost_cny": 0.0,
        }
        try:
            admission_payload = series.copy()
            admission_payload["decision_at"] = series["decision_at_admission"]
            status_column = (
                "admission_status_admission"
                if "admission_status_admission" in series.index
                else "admission_status"
            )
            admission_payload["admission_status"] = series[status_column]
            admission = _make_admission(admission_payload)
            if admission.admission_status == "REJECTED":
                base["status"] = "ABSTAINED"
                base["failure_code"] = "A1_REJECTED"
                base["total_latency_ms"] = (time.perf_counter() - started) * 1000
                rows.append(base)
                continue
            feature_row = {
                column: series[column]
                for column in sample.columns
                if column not in identity_columns
                and column not in admission_columns
                and not column.endswith("_eligibility")
                and not column.endswith("_admission")
            }
            feature_row["origin_event_id"] = admission.origin_event_id
            feature_row["decision_at"] = admission.decision_at
            request = _make_request(
                admission=admission, feature_row=feature_row, run_id=run_id
            )
            event_run_id = f"{run_id}::{pair_id}"
            plan = fixed_offline_plan(
                run_id=event_run_id,
                numeric_predictor_ref=predictor.model_id,
                post_freeze=True,
            )
            result = run_offline_graph(
                plan=plan,
                request=request,
                feature_row=feature_row,
                predict=predictor,
            )
            base.update(
                {
                    "status": "PASSED" if result.audit.passed else "FAILED",
                    "failure_code": None if result.audit.passed else "A5_AUDIT_FAILED",
                    "predicted_cu_g_l": result.prediction.predicted_cu_g_l,
                    "predicted_as_mg_l": result.prediction.predicted_as_mg_l,
                    "mode_code": result.process_mode.mode_code,
                    "mode_confidence": result.process_mode.confidence,
                    "mode_warnings": _json_cell(result.process_mode.warnings),
                    "audit_passed": result.audit.passed,
                    "audit_checks": _json_cell(result.audit.check_codes),
                    "total_latency_ms": result.resources.total_latency_ms,
                    "total_calls": result.resources.total_calls,
                    "total_input_tokens": result.resources.total_input_tokens,
                    "total_output_tokens": result.resources.total_output_tokens,
                    "total_estimated_cost_cny": result.resources.total_estimated_cost_cny,
                }
            )
        except Exception as exc:  # 单事件 fail-closed，但保留全量审计行
            base["failure_code"] = f"{type(exc).__name__}:{str(exc)[:160]}"
            base["total_latency_ms"] = (time.perf_counter() - started) * 1000
        rows.append(base)
    predictions = pd.DataFrame(rows)
    if predictions["pair_id"].duplicated().any():
        raise ValueError("冻结预测 pair_id 不唯一")
    if FORBIDDEN_PREDICTION_COLUMNS & set(predictions.columns):
        raise ValueError("冻结预测在读真值前出现真值列")
    return predictions


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(y_pred - y_true)
    bias = y_pred - y_true
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2,
        "MEDAE": float(median_absolute_error(y_true, y_pred)),
        "P90AE": float(np.percentile(absolute_error, 90)),
        "BIAS": float(np.mean(bias)),
    }


def _strata_frames(
    evaluated: pd.DataFrame, protocol: ExternalEvaluationProtocolV1
) -> list[tuple[str, str, pd.DataFrame]]:
    frames: list[tuple[str, str, pd.DataFrame]] = [("OVERALL", "ALL", evaluated)]
    if "CALENDAR_MONTH" in protocol.fixed_strata:
        months = evaluated["decision_at"].dt.to_period("M").astype(str)
        for label in sorted(months.unique()):
            frames.append(("CALENDAR_MONTH", label, evaluated.loc[months == label]))
    if "LEAD_TIME_BIN" in protocol.fixed_strata:
        edges = [-np.inf, *protocol.lead_time_bin_edges_hours, np.inf]
        bins = pd.cut(
            evaluated["lead_to_recorded_hours"],
            bins=edges,
            labels=protocol.lead_time_bin_labels,
            right=True,
            include_lowest=True,
        )
        for label in protocol.lead_time_bin_labels:
            frames.append(("LEAD_TIME_BIN", label, evaluated.loc[bins == label]))
    if "PROCESS_MODE" in protocol.fixed_strata:
        modes = evaluated["mode_code"].fillna("NO_MODE_ABSTAINED_OR_FAILED").astype(str)
        for label in sorted(modes.unique()):
            frames.append(("PROCESS_MODE", label, evaluated.loc[modes == label]))
    if "COVERAGE" in protocol.fixed_strata:
        coverage = np.where(
            evaluated["coverage_ready_4_of_7"], "READY_4_OF_7", "NOT_READY_4_OF_7"
        )
        for label in ("READY_4_OF_7", "NOT_READY_4_OF_7"):
            frames.append(("COVERAGE", label, evaluated.loc[coverage == label]))
    return frames


def score_external_predictions(
    *,
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    protocol: ExternalEvaluationProtocolV1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """只负责已冻结预测的事后计分；不接收模型或特征。"""

    _require_columns(
        outcomes, EXPECTED_OUTCOME_COLUMNS, source="sealed outcome", exact=True
    )
    if len(outcomes) != protocol.expected_pair_count:
        raise ValueError("封存真值行数与预注册协议不一致")
    if outcomes["pair_id"].duplicated().any():
        raise ValueError("封存真值 pair_id 不唯一")
    for column in ("target_cu_g_l", "target_as_mg_l"):
        values = pd.to_numeric(outcomes[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"封存真值 {column} 含缺失或非有限数")
        outcomes[column] = values
    evaluated = predictions.merge(
        outcomes[["pair_id", "target_cu_g_l", "target_as_mg_l"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(evaluated) != protocol.expected_pair_count or not evaluated["_merge"].eq("both").all():
        raise ValueError("预测与封存真值未完成 1:1 匹配")
    evaluated = evaluated.drop(columns="_merge")
    evaluated["decision_at"] = pd.to_datetime(
        evaluated["decision_at"], errors="raise", format="mixed"
    )
    evaluated["cu_error"] = evaluated["predicted_cu_g_l"] - evaluated["target_cu_g_l"]
    evaluated["as_error"] = evaluated["predicted_as_mg_l"] - evaluated["target_as_mg_l"]
    evaluated["cu_absolute_error"] = evaluated["cu_error"].abs()
    evaluated["as_absolute_error"] = evaluated["as_error"].abs()

    metric_rows: list[dict[str, Any]] = []
    operational_rows: list[dict[str, Any]] = []
    for dimension, stratum, frame in _strata_frames(evaluated, protocol):
        passed = frame.loc[
            frame["status"].eq("PASSED")
            & frame["audit_passed"].eq(True)  # noqa: E712
        ]
        attempted = len(frame)
        operational_rows.append(
            {
                "dimension": dimension,
                "stratum": stratum,
                "attempted_count": attempted,
                "passed_count": int(len(passed)),
                "abstained_count": int(frame["status"].eq("ABSTAINED").sum()),
                "failed_count": int(frame["status"].eq("FAILED").sum()),
                "warning_event_count": int(
                    frame["mode_warnings"].fillna("[]").ne("[]").sum()
                ),
                "pass_rate": float(len(passed) / attempted) if attempted else np.nan,
                "abstention_rate": float(frame["status"].eq("ABSTAINED").mean())
                if attempted
                else np.nan,
                "failure_rate": float(frame["status"].eq("FAILED").mean())
                if attempted
                else np.nan,
                "p50_latency_ms": float(
                    pd.to_numeric(frame["total_latency_ms"], errors="coerce").quantile(0.5)
                )
                if attempted
                else np.nan,
                "p95_latency_ms": float(
                    pd.to_numeric(frame["total_latency_ms"], errors="coerce").quantile(0.95)
                )
                if attempted
                else np.nan,
            }
        )
        for target, prediction, display, unit in TARGETS:
            row: dict[str, Any] = {
                "dimension": dimension,
                "stratum": stratum,
                "target_name": target,
                "target_display_name": display,
                "unit": unit,
                "sample_count": int(len(passed)),
            }
            if len(passed) >= protocol.minimum_metric_sample_count:
                values = _metric_values(
                    passed[target].to_numpy(dtype=float),
                    passed[prediction].to_numpy(dtype=float),
                )
            else:
                values = {metric: np.nan for metric in protocol.primary_metrics}
            row.update({metric.lower(): values[metric] for metric in protocol.primary_metrics})
            row["interval_coverage"] = np.nan
            row["interval_status"] = protocol.prediction_interval_status
            metric_rows.append(row)
    return evaluated, pd.DataFrame(metric_rows), pd.DataFrame(operational_rows)


def _moving_block_bootstrap_ci(
    *,
    evaluated: pd.DataFrame,
    protocol: ExternalEvaluationProtocolV1,
) -> pd.DataFrame:
    passed = evaluated.loc[
        evaluated["status"].eq("PASSED") & evaluated["audit_passed"].eq(True)
    ].sort_values(["decision_at", "pair_id"], kind="mergesort")
    n = len(passed)
    if n < 2:
        return pd.DataFrame(
            columns=["target_name", "metric", "lower", "upper", "method", "replicates"]
        )
    config = protocol.bootstrap
    block = min(config.block_length_events, n)
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []
    for target, prediction, *_ in TARGETS:
        truth = passed[target].to_numpy(dtype=float)
        estimate = passed[prediction].to_numpy(dtype=float)
        distributions = {metric: [] for metric in protocol.primary_metrics}
        blocks_needed = math.ceil(n / block)
        for _ in range(config.replicates):
            starts = rng.integers(0, n, size=blocks_needed)
            indices = np.concatenate(
                [np.mod(np.arange(start, start + block), n) for start in starts]
            )[:n]
            values = _metric_values(truth[indices], estimate[indices])
            for metric in protocol.primary_metrics:
                if np.isfinite(values[metric]):
                    distributions[metric].append(values[metric])
        for metric in protocol.primary_metrics:
            values = np.asarray(distributions[metric], dtype=float)
            rows.append(
                {
                    "target_name": target,
                    "metric": metric,
                    "lower": float(np.percentile(values, config.lower_percentile))
                    if len(values)
                    else np.nan,
                    "upper": float(np.percentile(values, config.upper_percentile))
                    if len(values)
                    else np.nan,
                    "method": config.method,
                    "block_length_events": block,
                    "replicates": config.replicates,
                    "random_seed": config.random_seed,
                }
            )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_report(
    *, metrics: pd.DataFrame, operational: pd.DataFrame, protocol: ExternalEvaluationProtocolV1
) -> str:
    overall = metrics.loc[metrics["dimension"].eq("OVERALL")]
    operations = operational.loc[operational["dimension"].eq("OVERALL")].iloc[0]
    lines = [
        "# 2026 外部时间留出一次性评价报告 V1",
        "",
        f"- 预注册样本：{protocol.expected_pair_count} 个相邻结果对。",
        f"- 成功预测：{int(operations['passed_count'])}。",
        f"- 弃权：{int(operations['abstained_count'])}。",
        f"- 失败：{int(operations['failed_count'])}。",
        "- 数值模型、特征、分层与指标均在真值读取前冻结。",
        "- 数值指标仅在 PASSED 且 A5 审计通过的事件上计算；全部 682 对同时报告失败与弃权率。",
        "- 当前无冻结区间模型，区间覆盖率记为 NA，未事后构造。",
        "",
        "## 整体指标",
        "",
        "| 目标 | n | MAE | RMSE | R² | MedAE | P90AE | Bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.target_display_name} | {int(row.sample_count)} | "
            f"{row.mae:.6g} | {row.rmse:.6g} | {row.r2:.6g} | "
            f"{row.medae:.6g} | {row.p90ae:.6g} | {row.bias:.6g} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "2026 与 2024—2025 存在已记录的协变量漂移；这是跨工况的外部时间检验，不是 IID 随机测试。",
            "",
        ]
    )
    return "\n".join(lines)


def header_only_real_preflight(
    *,
    project_root: str | Path,
    gate_path: str | Path,
    review_decision_path: str | Path,
    protocol_path: str | Path,
    external_dir: str | Path,
) -> dict[str, Any]:
    """只读真实工件表头和非真值清单，不 claim、不读真值行。"""

    root = Path(project_root).resolve()
    gate_file = Path(gate_path).resolve()
    external = Path(external_dir).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_external_evaluation_protocol(protocol_file)
    gate = load_external_release_gate(gate_file)
    manifest_path = root / protocol.external_build_manifest_path
    manifest_hash_ok = (
        sha256_file(manifest_path).lower()
        == protocol.external_build_manifest_sha256.lower()
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    header_requirements: dict[str, tuple[set[str], bool]] = {
        "core_feature_matrix_v2.csv": (
            {"origin_event_id", "decision_at", "origin_cu_g_l", "origin_as_mg_l"},
            False,
        ),
        "as_of_admission_card_core_v2.csv": (EXPECTED_ADMISSION_COLUMNS, False),
        "evaluation_eligibility_ledger_v2.csv": (EXPECTED_ELIGIBILITY_COLUMNS, False),
        "external_evaluation_index_v2.csv": (EXPECTED_INDEX_COLUMNS, False),
        "sealed/outcome_ledger_v2.csv": (set(EXPECTED_OUTCOME_COLUMNS), True),
    }
    header_checks: dict[str, bool] = {}
    for relative, (required, exact) in header_requirements.items():
        header = pd.read_csv(external / relative, nrows=0)
        actual = set(header.columns)
        header_checks[relative] = actual == required if exact else required <= actual

    freezes = gate.required_freezes

    def _resolve_gate_path(value: str | None) -> Path:
        raw = Path(str(value or ""))
        return raw.resolve() if raw.is_absolute() else (root / raw).resolve()

    def _bound_file(
        value: str | None,
        expected_sha256: str | None,
        *,
        required_path: Path | None = None,
    ) -> bool:
        path = _resolve_gate_path(value)
        return bool(
            value
            and expected_sha256
            and path.is_file()
            and (required_path is None or path == required_path.resolve())
            and sha256_file(path).lower() == expected_sha256.lower()
        )

    selected = root / str(freezes.selected_model_manifest_path or "")
    graph = root / freezes.agent_graph_path
    review = Path(review_decision_path).resolve()
    review_payload = json.loads(review.read_text(encoding="utf-8"))
    workbook = Path(str(review_payload.get("source_workbook") or "")).resolve()
    review_decision_bound = _bound_file(
        freezes.human_review_decision_path,
        freezes.human_review_decision_sha256,
        required_path=review,
    )
    review_ok = (
        review_payload.get("completion_status") == "COMPLETED_AND_ACCEPTED"
        and review_payload.get("completed") is True
        and review_payload.get("accepted_for_external_release") is True
        and review_decision_bound
        and workbook.is_file()
        and sha256_file(workbook).lower()
        == str(review_payload.get("source_workbook_sha256") or "").lower()
        and bool(freezes.human_review_workbook_sha256)
        and sha256_file(workbook).lower()
        == str(freezes.human_review_workbook_sha256 or "").lower()
    )
    selected_hash_ok = bool(
        selected.is_file()
        and freezes.selected_model_manifest_sha256
        and sha256_file(selected).lower()
        == freezes.selected_model_manifest_sha256.lower()
    )
    graph_hash_ok = bool(
        graph.is_file()
        and freezes.frozen_agent_graph_sha256
        and sha256_file(graph).lower() == freezes.frozen_agent_graph_sha256.lower()
    )
    timing_bound = _bound_file(
        freezes.timing_contract_v2_path,
        freezes.timing_contract_v2_sha256,
    )
    core_contract_bound = _bound_file(
        freezes.core_feature_contract_path,
        freezes.core_feature_contract_sha256,
    )
    model_selection_protocol_bound = _bound_file(
        freezes.model_selection_protocol_path,
        freezes.model_selection_protocol_sha256,
    )
    protocol_bound = _bound_file(
        freezes.external_evaluation_protocol_path,
        freezes.external_evaluation_protocol_sha256,
        required_path=protocol_file,
    )
    evaluator_bound = _bound_file(
        freezes.external_evaluator_path,
        freezes.external_evaluator_sha256,
        required_path=Path(__file__).resolve(),
    )
    runtime_manifest_bound = False
    runtime_manifest_hash: str | None = None
    runtime_verified_file_count = 0
    try:
        runtime_evidence = verify_runtime_freeze_manifest(
            project_root=root,
            manifest_path=freezes.external_runtime_manifest_path or "",
            expected_sha256=freezes.external_runtime_manifest_sha256 or "",
        )
    except (ExternalEvaluationTransactionError, OSError, ValueError, TypeError):
        pass
    else:
        runtime_manifest_bound = True
        runtime_manifest_hash = str(runtime_evidence["manifest_sha256"])
        runtime_verified_file_count = int(runtime_evidence["verified_file_count"])
    result = {
        "preflight_id": "EXTERNAL_2026_HEADER_ONLY_PREFLIGHT_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_evaluation_started": False,
        "claim_created": False,
        "sealed_outcome_rows_read": 0,
        "gate_sha256": sha256_file(gate_file),
        "protocol_sha256": sha256_file(protocol_file),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "gate_status": gate.status,
        "expected_event_count": protocol.expected_event_count,
        "expected_pair_count": protocol.expected_pair_count,
        "manifest_declared_event_count": int(
            (manifest.get("counts") or {}).get("target_events", -1)
        ),
        "manifest_declared_pair_count": int(
            (manifest.get("counts") or {}).get("strict_adjacent_pairs", -1)
        ),
        "manifest_hash_ok": manifest_hash_ok,
        "selected_model_hash_ok": selected_hash_ok,
        "agent_graph_hash_ok": graph_hash_ok,
        "manual_review_and_workbook_hash_ok": review_ok,
        "manual_review_decision_hash_bound_in_gate": review_decision_bound,
        "timing_contract_hash_bound_in_gate": timing_bound,
        "core_feature_contract_hash_bound_in_gate": core_contract_bound,
        "model_selection_protocol_hash_bound_in_gate": (
            model_selection_protocol_bound
        ),
        "protocol_hash_bound_in_gate": protocol_bound,
        "evaluator_code_hash_bound_in_gate": evaluator_bound,
        "runtime_manifest_hash_bound_in_gate": runtime_manifest_bound,
        "runtime_manifest_sha256": runtime_manifest_hash,
        "runtime_verified_file_count": runtime_verified_file_count,
        "header_checks": header_checks,
        "sealed_outcome_actual_sha256_computed": False,
    }
    result["ready_for_release_after_gate_binding"] = all(
        [
            manifest_hash_ok,
            selected_hash_ok,
            graph_hash_ok,
            review_ok,
            timing_bound,
            core_contract_bound,
            model_selection_protocol_bound,
            protocol_bound,
            evaluator_bound,
            runtime_manifest_bound,
            all(header_checks.values()),
        ]
    )
    result["ready_for_formal_run_now"] = bool(
        result["ready_for_release_after_gate_binding"]
        and gate.status == "RELEASED_FOR_SINGLE_EVALUATION"
        and bool(gate.release_evidence_path)
        and bool(gate.release_evidence_sha256)
    )
    return result


def run_external_evaluation_once(
    *, paths: ExternalEvaluationPaths, run_id: str
) -> dict[str, Any]:
    """执行一次性外部评价。调用者不能覆盖样本数或指标。"""

    live = os.environ.get("COPPER_MAS_LIVE_CALLS", "false").strip().lower()
    if live in {"1", "true", "yes", "on"}:
        raise ValueError("外部数值评价强制禁用 LLM/网络调用")
    protocol = load_external_evaluation_protocol(paths.protocol_path)
    if paths.output_dir.exists():
        raise FileExistsError(f"外部评价输出已存在: {paths.output_dir}")
    staging = paths.output_dir.parent / f".{paths.output_dir.name}.{run_id}.staging"
    if staging.exists():
        raise FileExistsError(f"外部评价临时目录已存在: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    claimed = False
    sealed_hash = ""
    started = time.perf_counter()
    try:
        claim_external_evaluation(
            project_root=paths.project_root,
            gate_path=paths.gate_path,
            claim_path=paths.claim_path,
            review_decision_path=paths.review_decision_path,
            run_id=run_id,
        )
        claimed = True
        gate = load_external_release_gate(paths.gate_path)
        manifest, sample = _load_non_outcome_inputs(paths=paths, protocol=protocol)
        selected_path = paths.project_root / str(
            gate.required_freezes.selected_model_manifest_path
        )
        predictions = _run_frozen_predictions(
            sample=sample,
            selected_manifest_path=selected_path,
            run_id=run_id,
        )
        if len(predictions) != protocol.expected_pair_count:
            raise ValueError("冻结预测行数不是预注册的 682")
        prediction_path = staging / "predictions_frozen_v1.csv"
        predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        mark_predictions_frozen(
            gate_path=paths.gate_path,
            claim_path=paths.claim_path,
            run_id=run_id,
            predictions_path=prediction_path,
            expected_rows=len(predictions),
        )

        # 从这一行状态转移之后，才允许对封存文件做任何正文读取或哈希。
        mark_outcome_access_started(
            gate_path=paths.gate_path, claim_path=paths.claim_path, run_id=run_id
        )
        assert_sealed_access_authorized(
            gate_path=paths.gate_path, claim_path=paths.claim_path, run_id=run_id
        )
        outcome_path = paths.external_dir / "sealed/outcome_ledger_v2.csv"
        sealed_hash = sha256_file(outcome_path)
        expected_sealed_hash = str(
            ((manifest.get("artifacts") or {}).get("sealed/outcome_ledger_v2.csv") or {}).get(
                "sha256", ""
            )
        )
        if not expected_sealed_hash or sealed_hash.lower() != expected_sealed_hash.lower():
            raise ValueError("封存真值 SHA-256 与构建清单不一致")
        outcomes = pd.read_csv(outcome_path, low_memory=False)
        evaluated, metrics, operational = score_external_predictions(
            predictions=predictions, outcomes=outcomes, protocol=protocol
        )
        confidence_intervals = _moving_block_bootstrap_ci(
            evaluated=evaluated, protocol=protocol
        )

        csv_outputs = {
            "external_event_prediction_audit_v1.csv": evaluated,
            "external_metrics_v1.csv": metrics,
            "external_metric_confidence_intervals_v1.csv": confidence_intervals,
            "external_operational_strata_v1.csv": operational,
        }
        for filename, frame in csv_outputs.items():
            frame.to_csv(staging / filename, index=False, encoding="utf-8-sig")
        report_path = staging / "2026外部时间留出一次性评价报告_V1.md"
        report_path.write_text(
            _render_report(metrics=metrics, operational=operational, protocol=protocol),
            encoding="utf-8",
        )

        primary = [prediction_path, *(staging / name for name in csv_outputs), report_path]
        hash_rows = [
            {"artifact": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in primary
        ]
        hash_frame = pd.DataFrame(hash_rows).sort_values("artifact", kind="mergesort")
        hash_path = staging / "artifact_hashes_v1.csv"
        hash_frame.to_csv(hash_path, index=False, encoding="utf-8-sig")
        overall_operations = operational.loc[
            operational["dimension"].eq("OVERALL")
        ].iloc[0]
        run_manifest = {
            "run_id": run_id,
            "protocol_id": protocol.protocol_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "partition_role": protocol.partition_role,
            "expected_pair_count": protocol.expected_pair_count,
            "attempted_count": int(overall_operations["attempted_count"]),
            "passed_count": int(overall_operations["passed_count"]),
            "abstained_count": int(overall_operations["abstained_count"]),
            "failed_count": int(overall_operations["failed_count"]),
            "frozen_model_id": str(predictions["frozen_model_id"].iloc[0]),
            "model_changes_after_outcome_access": 0,
            "strata_changes_after_outcome_access": 0,
            "llm_calls_made": 0,
            "network_calls_made": 0,
            "prediction_interval_status": protocol.prediction_interval_status,
            "interval_coverage": None,
            "metric_population": protocol.metric_population,
            "operational_population": protocol.operational_population,
            "outcome_missing_policy": protocol.outcome_missing_policy,
            "claim_path": str(paths.claim_path.resolve()),
            "protocol_sha256": sha256_file(paths.protocol_path),
            "external_build_manifest_sha256": protocol.external_build_manifest_sha256,
            "sealed_outcome_sha256": sealed_hash,
            "output_sha256": {
                **{row["artifact"]: row["sha256"] for row in hash_rows},
                hash_path.name: sha256_file(hash_path),
            },
        }
        run_manifest_path = staging / "run_manifest_v1.json"
        _write_json(run_manifest_path, run_manifest)
        paths.output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, paths.output_dir)
        final_manifest = paths.output_dir / "run_manifest_v1.json"
        complete_external_evaluation(
            gate_path=paths.gate_path,
            claim_path=paths.claim_path,
            run_id=run_id,
            result_manifest_path=final_manifest,
            sealed_outcome_sha256=sealed_hash,
        )
        return run_manifest
    except Exception as exc:
        if claimed:
            try:
                fail_external_evaluation_closed(
                    gate_path=paths.gate_path,
                    claim_path=paths.claim_path,
                    run_id=run_id,
                    exception=exc,
                )
            except Exception:
                pass
        raise
    finally:
        # 正式失败时保留 staging 用于同 run_id 人工恢复；仅清理空目录。
        if staging.exists() and not any(staging.iterdir()):
            shutil.rmtree(staging)
