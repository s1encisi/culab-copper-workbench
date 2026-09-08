"""2024--2025 开发期与 2026 外部期的过程输入协变量漂移诊断。

本模块刻意不接触 outcome ledger，也不把当前废铜液 Cu/As 放进漂移统计。
它回答的是“两个时期的过程输入 X 有多容易区分”，不是“模型在 2026 的误差是多少”。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


RUN_ID = "P2_COVARIATE_SHIFT_V1"
IDENTITY_COLUMNS = ("origin_event_id", "decision_at")
QUALITY_COLUMN = "origin_result_quality_eligible"
FORBIDDEN_PREFIXES = ("target_", "next_", "lead_")
PROTECTED_CURRENT_RESULT_BASES = (
    "origin_cu",
    "origin_as",
    "current_cu",
    "current_as",
    "present_cu",
    "present_as",
    "raw_tank_cu",
    "raw_tank_as",
    "waste_tank_cu",
    "waste_tank_as",
)
PROCESS_FEATURE_PATTERN = re.compile(
    r"^(?P<base>[a-z0-9_]+)__(?:"
    r"t_minus_[0-9]+h(?:__(?:missing|age_h))?"
    r"|mean_12h|std_12h|range_12h|valid_count_12h|slope_per_h_12h)$"
)


class SecurityBoundaryError(ValueError):
    """输入表违反外部评价保护边界。"""


@dataclass(frozen=True)
class HeaderAudit:
    development_header: tuple[str, ...]
    external_header: tuple[str, ...]
    development_process_features: tuple[str, ...]
    external_process_features: tuple[str, ...]
    common_process_features: tuple[str, ...]
    development_only_process_features: tuple[str, ...]
    external_only_process_features: tuple[str, ...]
    protected_columns_excluded: tuple[str, ...]
    development_common_order_matches: bool
    external_common_order_matches: bool
    development_header_sha256: str
    external_header_sha256: str
    common_feature_order_sha256: str


@dataclass(frozen=True)
class SafeViewResult:
    path: Path
    row_count: int
    feature_count: int
    feature_order_sha256: str
    selected_matrix_sha256: str
    excluded_protected_columns: tuple[str, ...]
    forbidden_columns: tuple[str, ...]


@dataclass(frozen=True)
class CovariateShiftResult:
    output_dir: Path
    feature_metrics: pd.DataFrame
    base_signal_summary: pd.DataFrame
    domain_fold_metrics: pd.DataFrame
    domain_summary: pd.DataFrame
    manifest: Mapping[str, Any]


def _sha256_text(items: Iterable[str]) -> str:
    payload = "\n".join(items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_header(path: Path) -> tuple[str, ...]:
    """只读 CSV 第一行；任何数值载入之前先完成列级安全检查。"""

    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = tuple(next(csv.reader(handle)))
        except StopIteration as exc:
            raise ValueError(f"空 CSV：{path}") from exc
    if not header or any(not column for column in header):
        raise ValueError(f"CSV 表头为空或含空列名：{path}")
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise ValueError(f"CSV 表头含重复列：{duplicates}")
    return header


def _column_tokens(column: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"__|\.", column.lower().strip()) if part)


def is_forbidden_future_or_target_column(column: str) -> bool:
    """拒绝显式 target/next/lead 字段，包含派生列中的同名 token。"""

    return any(
        token.startswith(FORBIDDEN_PREFIXES)
        for token in _column_tokens(column)
    )


def is_protected_current_result_column(column: str) -> bool:
    """识别当前废铜液 Cu/As 及其直接派生或近似命名。"""

    normalized = column.lower().strip()
    base = normalized.split("__", 1)[0]
    return any(
        base == protected or base.startswith(f"{protected}_")
        for protected in PROTECTED_CURRENT_RESULT_BASES
    )


def is_process_feature(column: str) -> bool:
    return PROCESS_FEATURE_PATTERN.fullmatch(column.lower().strip()) is not None


def _process_feature_names(header: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        column
        for column in header
        if is_process_feature(column)
        and not is_protected_current_result_column(column)
    )


def _raise_on_forbidden_columns(*headers: Sequence[str]) -> None:
    forbidden = sorted(
        {
            column
            for header in headers
            for column in header
            if is_forbidden_future_or_target_column(column)
        }
    )
    if forbidden:
        raise SecurityBoundaryError(
            "core_feature_matrix 含禁止的 target_/next_/lead_ 列；"
            f"在读取数值前已拒绝：{forbidden}"
        )


def audit_headers(development_path: Path, external_path: Path) -> HeaderAudit:
    development_header = read_csv_header(development_path)
    external_header = read_csv_header(external_path)
    _raise_on_forbidden_columns(development_header, external_header)

    development_features = _process_feature_names(development_header)
    external_features = _process_feature_names(external_header)
    external_set = set(external_features)
    development_set = set(development_features)
    common = tuple(feature for feature in development_features if feature in external_set)
    external_common = tuple(feature for feature in external_features if feature in development_set)
    protected = tuple(
        sorted(
            {
                column
                for header in (development_header, external_header)
                for column in header
                if is_protected_current_result_column(column)
            }
        )
    )
    if not common:
        raise ValueError("两份输入没有共同的冻结过程特征。")
    return HeaderAudit(
        development_header=development_header,
        external_header=external_header,
        development_process_features=development_features,
        external_process_features=external_features,
        common_process_features=common,
        development_only_process_features=tuple(
            feature for feature in development_features if feature not in external_set
        ),
        external_only_process_features=tuple(
            feature for feature in external_features if feature not in development_set
        ),
        protected_columns_excluded=protected,
        development_common_order_matches=tuple(
            feature for feature in development_features if feature in set(common)
        )
        == common,
        external_common_order_matches=external_common == common,
        development_header_sha256=_sha256_text(development_header),
        external_header_sha256=_sha256_text(external_header),
        common_feature_order_sha256=_sha256_text(common),
    )


def _selected_frame_sha256(frame: pd.DataFrame) -> str:
    """仅对已经通过 usecols 读取的安全字段形成内容指纹。"""

    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.columns)).encode("utf-8"))
    hashed_rows = pd.util.hash_pandas_object(frame, index=True, categorize=False)
    digest.update(hashed_rows.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _read_selected_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """显式 usecols；禁止退化为读取整表后再 drop。"""

    header = read_csv_header(path)
    _raise_on_forbidden_columns(header)
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"输入缺少所需安全字段：{missing}")
    selected = pd.read_csv(
        path,
        usecols=list(columns),
        encoding="utf-8-sig",
        low_memory=False,
    )
    return selected.loc[:, list(columns)]


def materialize_safe_process_view(
    source_path: Path,
    safe_view_path: Path,
    *,
    expected_feature_count: int | None = None,
) -> SafeViewResult:
    """从完整矩阵生成不含 Cu/As、质量标志和任何标签的安全过程视图。"""

    source_path = Path(source_path).resolve()
    safe_view_path = Path(safe_view_path).resolve()
    header = read_csv_header(source_path)
    _raise_on_forbidden_columns(header)
    features = _process_feature_names(header)
    if expected_feature_count is not None and len(features) != expected_feature_count:
        raise ValueError(
            f"过程特征数应为 {expected_feature_count}，实际为 {len(features)}。"
        )
    missing_identity = [column for column in IDENTITY_COLUMNS if column not in header]
    if missing_identity:
        raise ValueError(f"完整矩阵缺少安全视图主键/时间字段：{missing_identity}")
    selected_columns = (*IDENTITY_COLUMNS, *features)
    safe_frame = _read_selected_columns(source_path, selected_columns)
    forbidden_selected = tuple(
        column
        for column in safe_frame.columns
        if is_forbidden_future_or_target_column(column)
    )
    protected_selected = tuple(
        column
        for column in safe_frame.columns
        if is_protected_current_result_column(column) or column == QUALITY_COLUMN
    )
    if forbidden_selected or protected_selected:
        raise SecurityBoundaryError(
            "安全视图仍含禁止字段："
            f"future_or_target={forbidden_selected}, protected={protected_selected}"
        )
    safe_view_path.parent.mkdir(parents=True, exist_ok=True)
    safe_frame.to_csv(
        safe_view_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    written_header = read_csv_header(safe_view_path)
    _raise_on_forbidden_columns(written_header)
    if tuple(written_header) != selected_columns:
        raise RuntimeError("安全视图写出后的字段顺序与冻结顺序不一致。")
    protected_written = tuple(
        column
        for column in written_header
        if is_protected_current_result_column(column) or column == QUALITY_COLUMN
    )
    if protected_written:
        raise SecurityBoundaryError(f"安全视图写出后发现保护字段：{protected_written}")
    return SafeViewResult(
        path=safe_view_path,
        row_count=len(safe_frame),
        feature_count=len(features),
        feature_order_sha256=_sha256_text(features),
        selected_matrix_sha256=_selected_frame_sha256(safe_frame),
        excluded_protected_columns=tuple(
            column
            for column in header
            if is_protected_current_result_column(column) or column == QUALITY_COLUMN
        ),
        forbidden_columns=forbidden_selected,
    )


def _load_numeric_features(path: Path, feature_columns: Sequence[str]) -> pd.DataFrame:
    selected = _read_selected_columns(path, feature_columns)
    converted_columns: dict[str, pd.Series] = {}
    invalid: dict[str, int] = {}
    for column in feature_columns:
        converted = pd.to_numeric(selected[column], errors="coerce")
        bad_count = int((selected[column].notna() & converted.isna()).sum())
        if bad_count:
            invalid[column] = bad_count
        converted_columns[column] = converted.astype(float)
    if invalid:
        raise ValueError(f"过程特征中存在不可解析的非数值：{invalid}")
    numeric = pd.DataFrame(converted_columns, index=selected.index)
    return numeric.loc[:, list(feature_columns)]


def _load_identity_time(path: Path) -> pd.Series:
    """只读取安全主键与决策时间；不读取当前结果、质量标志或未来标签。"""

    selected = _read_selected_columns(path, IDENTITY_COLUMNS)
    if selected["origin_event_id"].isna().any():
        raise ValueError("origin_event_id 含空值，无法执行时间块域诊断。")
    parsed = pd.to_datetime(selected["decision_at"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("decision_at 含不可解析值，无法执行时间块域诊断。")
    return parsed.reset_index(drop=True)


def _safe_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    output = float(value)
    return output if math.isfinite(output) else None


def _fit_development_psi_bins(
    development_nonmissing: np.ndarray,
    requested_bins: int,
) -> np.ndarray:
    if development_nonmissing.size == 0:
        return np.asarray([-np.inf, np.inf], dtype=float)
    quantiles = np.linspace(0.0, 1.0, requested_bins + 1)[1:-1]
    cutpoints = np.unique(np.quantile(development_nonmissing, quantiles))
    cutpoints = cutpoints[np.isfinite(cutpoints)]
    return np.concatenate(([-np.inf], cutpoints, [np.inf])).astype(float)


def _psi_from_development_bins(
    development: pd.Series,
    external: pd.Series,
    *,
    requested_bins: int,
    epsilon: float,
) -> tuple[float, float, dict[str, Any]]:
    development_values = development.dropna().to_numpy(dtype=float)
    external_values = external.dropna().to_numpy(dtype=float)
    edges = _fit_development_psi_bins(development_values, requested_bins)
    development_nonmissing_counts = np.histogram(
        development_values, bins=edges
    )[0].astype(float)
    external_nonmissing_counts = np.histogram(external_values, bins=edges)[0].astype(float)
    development_counts_with_missing = np.append(
        development_nonmissing_counts, float(development.isna().sum())
    )
    external_counts_with_missing = np.append(
        external_nonmissing_counts, float(external.isna().sum())
    )
    bin_count_with_missing = len(development_counts_with_missing)
    development_prob = (development_counts_with_missing + epsilon) / (
        development_counts_with_missing.sum() + epsilon * bin_count_with_missing
    )
    external_prob = (external_counts_with_missing + epsilon) / (
        external_counts_with_missing.sum() + epsilon * bin_count_with_missing
    )
    psi_including_missing = float(
        np.sum((external_prob - development_prob) * np.log(external_prob / development_prob))
    )
    if development_values.size and external_values.size:
        nonmissing_bin_count = len(development_nonmissing_counts)
        development_nonmissing_prob = (
            development_nonmissing_counts + epsilon
        ) / (
            development_nonmissing_counts.sum() + epsilon * nonmissing_bin_count
        )
        external_nonmissing_prob = (external_nonmissing_counts + epsilon) / (
            external_nonmissing_counts.sum() + epsilon * nonmissing_bin_count
        )
        psi_nonmissing_only = float(
            np.sum(
                (external_nonmissing_prob - development_nonmissing_prob)
                * np.log(external_nonmissing_prob / development_nonmissing_prob)
            )
        )
        nonmissing_status = "OK"
    else:
        psi_nonmissing_only = np.nan
        nonmissing_status = "ONE_OR_BOTH_PERIODS_HAVE_NO_NONMISSING_VALUE"
    detail = {
        "fitted_on": "development_2024_2025_only",
        "requested_quantile_bins": requested_bins,
        "finite_cutpoints": [float(value) for value in edges[1:-1]],
        "nonmissing_interval_count": int(len(edges) - 1),
        "missing_bin_included": True,
        "development_nonmissing_counts": development_nonmissing_counts.astype(int).tolist(),
        "external_nonmissing_counts": external_nonmissing_counts.astype(int).tolist(),
        "development_counts_with_missing_bin": development_counts_with_missing.astype(int).tolist(),
        "external_counts_with_missing_bin": external_counts_with_missing.astype(int).tolist(),
        "psi_including_missing_bin": psi_including_missing,
        "psi_nonmissing_only": _safe_float(psi_nonmissing_only),
        "psi_nonmissing_only_status": nonmissing_status,
        "epsilon_count_smoothing": epsilon,
    }
    return psi_including_missing, psi_nonmissing_only, detail


def _feature_kind(feature: str) -> str:
    if feature.endswith("__missing"):
        return "missing_indicator"
    if feature.endswith("__age_h"):
        return "observation_age"
    suffix = feature.split("__", 1)[1]
    if suffix.startswith("t_minus_"):
        return "anchor_value"
    if suffix.startswith("valid_count_"):
        return "window_valid_count"
    if suffix.startswith("slope_"):
        return "window_slope"
    return "window_statistic"


def _shift_role(feature_kind: str) -> str:
    if feature_kind in {
        "missing_indicator",
        "observation_age",
        "window_valid_count",
    }:
        return "MEASUREMENT_AVAILABILITY"
    return "NONMISSING_NUMERIC_STATE"


def _process_group(base_signal: str) -> str:
    if base_signal.startswith("phase1_"):
        return "一期"
    if base_signal.startswith("phase2_"):
        return "二期"
    if base_signal.startswith("stage3_"):
        return "三段"
    if base_signal.startswith("stage4_"):
        return "四段"
    if base_signal.startswith("stage34_"):
        return "三四段公共"
    return "其他"


def calculate_feature_shift_metrics(
    development: pd.DataFrame,
    external: pd.DataFrame,
    *,
    base_signal_chinese_names: Mapping[str, str] | None = None,
    psi_bins: int = 10,
    psi_epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if tuple(development.columns) != tuple(external.columns):
        raise ValueError("开发期与外部期特征列或顺序不一致。")
    chinese_names = dict(base_signal_chinese_names or {})
    rows: list[dict[str, Any]] = []
    bin_details: dict[str, Any] = {}
    for order, feature in enumerate(development.columns):
        development_series = development[feature]
        external_series = external[feature]
        development_nonmissing = development_series.dropna()
        external_nonmissing = external_series.dropna()
        development_count = len(development_series)
        external_count = len(external_series)
        development_missing_rate = float(development_series.isna().mean())
        external_missing_rate = float(external_series.isna().mean())

        development_median = (
            float(development_nonmissing.median()) if len(development_nonmissing) else np.nan
        )
        external_median = (
            float(external_nonmissing.median()) if len(external_nonmissing) else np.nan
        )
        development_q1 = (
            float(development_nonmissing.quantile(0.25)) if len(development_nonmissing) else np.nan
        )
        development_q3 = (
            float(development_nonmissing.quantile(0.75)) if len(development_nonmissing) else np.nan
        )
        development_iqr = development_q3 - development_q1
        raw_median_difference = external_median - development_median
        if math.isfinite(development_iqr) and development_iqr > 0:
            robust_shift = raw_median_difference / development_iqr
            robust_status = "OK"
        elif math.isfinite(raw_median_difference) and raw_median_difference == 0:
            robust_shift = 0.0
            robust_status = "DEV_IQR_ZERO_EQUAL_MEDIAN"
        else:
            robust_shift = np.nan
            robust_status = "DEV_IQR_ZERO_OR_UNAVAILABLE"

        development_mean = (
            float(development_nonmissing.mean()) if len(development_nonmissing) else np.nan
        )
        external_mean = float(external_nonmissing.mean()) if len(external_nonmissing) else np.nan
        development_std = (
            float(development_nonmissing.std(ddof=1)) if len(development_nonmissing) > 1 else np.nan
        )
        external_std = (
            float(external_nonmissing.std(ddof=1)) if len(external_nonmissing) > 1 else np.nan
        )
        pooled_sd = math.sqrt((development_std**2 + external_std**2) / 2.0) if (
            math.isfinite(development_std) and math.isfinite(external_std)
        ) else np.nan
        raw_mean_difference = external_mean - development_mean
        if math.isfinite(pooled_sd) and pooled_sd > 0:
            smd = raw_mean_difference / pooled_sd
            smd_status = "OK"
        elif math.isfinite(raw_mean_difference) and raw_mean_difference == 0:
            smd = 0.0
            smd_status = "POOLED_SD_ZERO_EQUAL_MEAN"
        else:
            smd = np.nan
            smd_status = "POOLED_SD_ZERO_OR_UNAVAILABLE"

        development_min = (
            float(development_nonmissing.min()) if len(development_nonmissing) else np.nan
        )
        development_max = (
            float(development_nonmissing.max()) if len(development_nonmissing) else np.nan
        )
        if len(external_nonmissing) and math.isfinite(development_min) and math.isfinite(development_max):
            outside_count = int(
                ((external_nonmissing < development_min) | (external_nonmissing > development_max)).sum()
            )
            outside_rate = outside_count / len(external_nonmissing)
        else:
            outside_count = 0
            outside_rate = np.nan

        psi_including_missing, psi_nonmissing_only, detail = _psi_from_development_bins(
            development_series,
            external_series,
            requested_bins=psi_bins,
            epsilon=psi_epsilon,
        )
        bin_details[feature] = detail
        base_signal = feature.split("__", 1)[0]
        feature_kind = _feature_kind(feature)
        rows.append(
            {
                "feature_order": order,
                "feature_name": feature,
                "base_signal": base_signal,
                "基础信号中文名": chinese_names.get(base_signal, ""),
                "process_group": _process_group(base_signal),
                "derived_feature_kind": feature_kind,
                "shift_role": _shift_role(feature_kind),
                "development_row_count": development_count,
                "external_row_count": external_count,
                "development_nonmissing_count": len(development_nonmissing),
                "external_nonmissing_count": len(external_nonmissing),
                "development_missing_rate": development_missing_rate,
                "external_missing_rate": external_missing_rate,
                "missing_rate_difference_external_minus_development": (
                    external_missing_rate - development_missing_rate
                ),
                "development_median": development_median,
                "external_median": external_median,
                "raw_median_difference": raw_median_difference,
                "development_iqr": development_iqr,
                "robust_median_shift_over_development_iqr": robust_shift,
                "robust_shift_status": robust_status,
                "development_mean": development_mean,
                "external_mean": external_mean,
                "raw_mean_difference": raw_mean_difference,
                "development_std": development_std,
                "external_std": external_std,
                "standardized_mean_difference": smd,
                "smd_status": smd_status,
                "psi_including_missing_bin_development_fitted": psi_including_missing,
                "psi_nonmissing_only_development_fitted_bins": psi_nonmissing_only,
                "psi_nonmissing_only_status": detail["psi_nonmissing_only_status"],
                "psi_nonmissing_interval_count": detail["nonmissing_interval_count"],
                "development_min": development_min,
                "development_max": development_max,
                "external_outside_development_range_count": outside_count,
                "external_outside_development_range_rate_nonmissing": outside_rate,
            }
        )
    return pd.DataFrame(rows), bin_details


def aggregate_by_base_signal(feature_metrics: pd.DataFrame) -> pd.DataFrame:
    """按基础信号分开测量可用性与非缺失数值漂移。"""

    rows: list[dict[str, Any]] = []
    for base_signal, group in feature_metrics.groupby("base_signal", sort=False):
        availability = group.loc[
            group["shift_role"] == "MEASUREMENT_AVAILABILITY"
        ]
        numeric = group.loc[
            group["shift_role"] == "NONMISSING_NUMERIC_STATE"
        ]
        anchor_numeric = numeric.loc[numeric["derived_feature_kind"] == "anchor_value"]
        abs_anchor_missing = anchor_numeric[
            "missing_rate_difference_external_minus_development"
        ].abs()
        numeric_conditional_psi = numeric[
            "psi_nonmissing_only_development_fitted_bins"
        ]
        numeric_abs_smd = numeric["standardized_mean_difference"].abs()
        numeric_outside = numeric[
            "external_outside_development_range_rate_nonmissing"
        ]
        availability_psi = availability[
            "psi_including_missing_bin_development_fitted"
        ]

        def _row(suffix: str) -> pd.Series | None:
            matches = group.loc[group["feature_name"] == f"{base_signal}__{suffix}"]
            return None if matches.empty else matches.iloc[0]

        t0 = _row("t_minus_0h")
        mean12 = _row("mean_12h")

        def _value(row: pd.Series | None, column: str) -> float:
            return np.nan if row is None else float(row[column])

        rows.append(
            {
                "base_signal": base_signal,
                "基础信号中文名": group["基础信号中文名"].iloc[0],
                "process_group": group["process_group"].iloc[0],
                "derived_feature_count": len(group),
                "availability_feature_count": len(availability),
                "numeric_state_feature_count": len(numeric),
                "t0_development_missing_rate": _value(t0, "development_missing_rate"),
                "t0_external_missing_rate": _value(t0, "external_missing_rate"),
                "t0_missing_rate_difference_external_minus_development": _value(
                    t0, "missing_rate_difference_external_minus_development"
                ),
                "mean12_development_missing_rate": _value(
                    mean12, "development_missing_rate"
                ),
                "mean12_external_missing_rate": _value(mean12, "external_missing_rate"),
                "mean12_missing_rate_difference_external_minus_development": _value(
                    mean12, "missing_rate_difference_external_minus_development"
                ),
                "median_abs_anchor_missing_rate_difference": abs_anchor_missing.median(),
                "max_abs_anchor_missing_rate_difference": abs_anchor_missing.max(),
                "median_availability_feature_psi": availability_psi.median(),
                "p90_availability_feature_psi": availability_psi.quantile(0.90),
                "max_availability_feature_psi": availability_psi.max(),
                "t0_nonmissing_psi": _value(
                    t0, "psi_nonmissing_only_development_fitted_bins"
                ),
                "t0_nonmissing_smd": _value(t0, "standardized_mean_difference"),
                "t0_external_outside_development_range_rate": _value(
                    t0, "external_outside_development_range_rate_nonmissing"
                ),
                "mean12_nonmissing_psi": _value(
                    mean12, "psi_nonmissing_only_development_fitted_bins"
                ),
                "mean12_nonmissing_smd": _value(
                    mean12, "standardized_mean_difference"
                ),
                "mean12_external_outside_development_range_rate": _value(
                    mean12, "external_outside_development_range_rate_nonmissing"
                ),
                "median_numeric_nonmissing_psi": numeric_conditional_psi.median(),
                "p90_numeric_nonmissing_psi": numeric_conditional_psi.quantile(0.90),
                "max_numeric_nonmissing_psi": numeric_conditional_psi.max(),
                "numeric_columns_nonmissing_psi_ge_0_10": int(
                    (numeric_conditional_psi >= 0.10).sum()
                ),
                "numeric_columns_nonmissing_psi_ge_0_25": int(
                    (numeric_conditional_psi >= 0.25).sum()
                ),
                "median_abs_numeric_smd": numeric_abs_smd.median(),
                "p90_abs_numeric_smd": numeric_abs_smd.quantile(0.90),
                "max_abs_numeric_smd": numeric_abs_smd.max(),
                "median_numeric_external_outside_development_range_rate": (
                    numeric_outside.median()
                ),
                "max_numeric_external_outside_development_range_rate": (
                    numeric_outside.max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _domain_pipeline(model_name: str, random_state: int) -> Pipeline:
    if model_name == "LogisticRegression":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scaler", RobustScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "HistGradientBoosting":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=100,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        early_stopping=False,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"未知域分类器：{model_name}")


def _domain_split_specs(
    development_time: pd.Series,
    external_time: pd.Series,
    domain: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    strategies: Sequence[str],
    purge_gap_hours: float,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    sample_index = np.arange(len(domain), dtype=int)
    if "random_stratified" in strategies:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        for fold_index, (train_index, validation_index) in enumerate(
            splitter.split(sample_index, domain), start=1
        ):
            specs.append(
                {
                    "strategy": "RANDOM_STRATIFIED",
                    "fold_index": fold_index,
                    "train_index": train_index,
                    "validation_index": validation_index,
                    "purge_gap_hours": 0.0,
                    "development_validation_start": None,
                    "development_validation_end": None,
                    "external_validation_start": None,
                    "external_validation_end": None,
                }
            )
    if "paired_purged_time_blocks" in strategies:
        if len(development_time) != int((domain == 0).sum()) or len(external_time) != int(
            (domain == 1).sum()
        ):
            raise ValueError("时间序列长度与域标签长度不一致。")
        development_order = np.argsort(
            development_time.to_numpy(dtype="datetime64[ns]"), kind="stable"
        )
        external_order_local = np.argsort(
            external_time.to_numpy(dtype="datetime64[ns]"), kind="stable"
        )
        development_blocks = np.array_split(development_order, n_splits)
        external_blocks_local = np.array_split(external_order_local, n_splits)
        external_offset = len(development_time)
        combined_time = pd.concat(
            [development_time, external_time], ignore_index=True
        )
        gap = pd.Timedelta(hours=purge_gap_hours)
        for fold_index, (development_block, external_block_local) in enumerate(
            zip(development_blocks, external_blocks_local, strict=True), start=1
        ):
            external_block = external_block_local + external_offset
            validation_index = np.concatenate(
                [development_block, external_block]
            ).astype(int)
            train_mask = np.ones(len(domain), dtype=bool)
            train_mask[validation_index] = False

            development_start = development_time.iloc[development_block].min()
            development_end = development_time.iloc[development_block].max()
            external_start = external_time.iloc[external_block_local].min()
            external_end = external_time.iloc[external_block_local].max()
            development_train_candidates = np.where(train_mask & (domain == 0))[0]
            external_train_candidates = np.where(train_mask & (domain == 1))[0]
            development_keep = ~combined_time.iloc[development_train_candidates].between(
                development_start - gap, development_end + gap, inclusive="both"
            ).to_numpy()
            external_keep = ~combined_time.iloc[external_train_candidates].between(
                external_start - gap, external_end + gap, inclusive="both"
            ).to_numpy()
            train_index = np.concatenate(
                [
                    development_train_candidates[development_keep],
                    external_train_candidates[external_keep],
                ]
            ).astype(int)
            specs.append(
                {
                    "strategy": "PAIRED_PURGED_CONTIGUOUS_TIME_BLOCK",
                    "fold_index": fold_index,
                    "train_index": train_index,
                    "validation_index": validation_index,
                    "purge_gap_hours": purge_gap_hours,
                    "development_validation_start": development_start,
                    "development_validation_end": development_end,
                    "external_validation_start": external_start,
                    "external_validation_end": external_end,
                }
            )
    unknown = sorted(
        set(strategies) - {"random_stratified", "paired_purged_time_blocks"}
    )
    if unknown:
        raise ValueError(f"未知域交叉验证策略：{unknown}")
    return specs


def run_domain_separability_diagnostic(
    development: pd.DataFrame,
    external: pd.DataFrame,
    *,
    development_time: pd.Series | None = None,
    external_time: pd.Series | None = None,
    model_names: Sequence[str] = ("LogisticRegression",),
    n_splits: int = 5,
    random_state: int = 20260826,
    cv_strategies: Sequence[str] = ("random_stratified",),
    purge_gap_hours: float = 24.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用 X 判别时期；仅为可分性诊断，绝不作为 A4 预测器或选模依据。"""

    if tuple(development.columns) != tuple(external.columns):
        raise ValueError("域分类器输入的特征列或顺序不一致。")
    combined = pd.concat([development, external], ignore_index=True)
    domain = np.concatenate(
        [np.zeros(len(development), dtype=int), np.ones(len(external), dtype=int)]
    )
    if min(np.bincount(domain)) < n_splits:
        raise ValueError("每个时期的样本数必须不少于域分类交叉验证折数。")
    if "paired_purged_time_blocks" in cv_strategies and (
        development_time is None or external_time is None
    ):
        raise ValueError("连续时间块域诊断需要两期 decision_at。")
    split_specs = _domain_split_specs(
        development_time
        if development_time is not None
        else pd.Series(pd.date_range("2000-01-01", periods=len(development), freq="h")),
        external_time
        if external_time is not None
        else pd.Series(pd.date_range("2001-01-01", periods=len(external), freq="h")),
        domain,
        n_splits=n_splits,
        random_state=random_state,
        strategies=cv_strategies,
        purge_gap_hours=purge_gap_hours,
    )
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for spec in split_specs:
            fold_index = int(spec["fold_index"])
            train_index = np.asarray(spec["train_index"], dtype=int)
            validation_index = np.asarray(spec["validation_index"], dtype=int)
            pipeline = _domain_pipeline(
                model_name,
                random_state
                + fold_index
                + (100 if spec["strategy"].startswith("PAIRED") else 0),
            )
            started = time.perf_counter()
            pipeline.fit(combined.iloc[train_index], domain[train_index])
            fit_seconds = time.perf_counter() - started
            probabilities = pipeline.predict_proba(combined.iloc[validation_index])[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            true_domain = domain[validation_index]
            rows.append(
                {
                    "run_id": RUN_ID,
                    "model_name": model_name,
                    "cv_strategy": spec["strategy"],
                    "fold_id": f"DOMAIN_FOLD_{fold_index}",
                    "train_sample_count": len(train_index),
                    "validation_sample_count": len(validation_index),
                    "train_development_count": int((domain[train_index] == 0).sum()),
                    "train_external_count": int((domain[train_index] == 1).sum()),
                    "validation_development_count": int((true_domain == 0).sum()),
                    "validation_external_count": int((true_domain == 1).sum()),
                    "roc_auc": roc_auc_score(true_domain, probabilities),
                    "balanced_accuracy": balanced_accuracy_score(true_domain, predictions),
                    "accuracy": accuracy_score(true_domain, predictions),
                    "fit_time_seconds": fit_seconds,
                    "preprocessing_fit_scope": "DOMAIN_TRAIN_FOLD_ONLY",
                    "purge_gap_hours": spec["purge_gap_hours"],
                    "development_validation_start": spec[
                        "development_validation_start"
                    ],
                    "development_validation_end": spec[
                        "development_validation_end"
                    ],
                    "external_validation_start": spec["external_validation_start"],
                    "external_validation_end": spec["external_validation_end"],
                }
            )
    fold_metrics = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (model_name, cv_strategy), group in fold_metrics.groupby(
        ["model_name", "cv_strategy"], sort=False
    ):
        summary_rows.append(
            {
                "run_id": RUN_ID,
                "model_name": model_name,
                "cv_strategy": cv_strategy,
                "fold_count": len(group),
                "roc_auc_mean": group["roc_auc"].mean(),
                "roc_auc_std_ddof1": group["roc_auc"].std(ddof=1),
                "balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                "balanced_accuracy_std_ddof1": group["balanced_accuracy"].std(ddof=1),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std_ddof1": group["accuracy"].std(ddof=1),
                "diagnostic_only": True,
                "not_a4_prediction_model": True,
            }
        )
    return fold_metrics, pd.DataFrame(summary_rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "—"


def _render_report(
    *,
    header_audit: HeaderAudit,
    development_rows: int,
    external_rows: int,
    feature_metrics: pd.DataFrame,
    base_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    domain_enabled: bool,
) -> str:
    psi_including_missing = feature_metrics[
        "psi_including_missing_bin_development_fitted"
    ]
    numeric_rows = feature_metrics.loc[
        feature_metrics["shift_role"] == "NONMISSING_NUMERIC_STATE"
    ]
    numeric_psi = numeric_rows["psi_nonmissing_only_development_fitted_bins"]
    numeric_abs_smd = numeric_rows["standardized_mean_difference"].abs()
    numeric_outside = numeric_rows[
        "external_outside_development_range_rate_nonmissing"
    ]
    ranked_availability = base_summary.sort_values(
        [
            "median_abs_anchor_missing_rate_difference",
            "max_abs_anchor_missing_rate_difference",
            "base_signal",
        ],
        ascending=[False, False, True],
        kind="mergesort",
    )
    ranked_numeric = base_summary.sort_values(
        ["median_numeric_nonmissing_psi", "p90_numeric_nonmissing_psi", "base_signal"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    top_availability = ranked_availability.iloc[0]
    top_numeric = ranked_numeric.head(3)
    lines = [
        "# 2024—2025 开发期与 2026 外部期过程输入协变量漂移审计（V1）",
        "",
        "## 1. 结论边界",
        "",
        (
            "本审计只比较预测时可用的过程输入 X。它没有读取 2026 Cu/As 当前值或下一次结果，"
            "没有读取任何 outcome ledger，也没有计算 2026 预测误差。"
        ),
        (
            "因此，本报告只能说明 2026 的过程输入与 2024—2025 是否存在可见协变量漂移；"
            "不能据此断言 A4 在 2026 的误差大小，更不能据此选择、修改或淘汰 A4。"
        ),
        "当前废铜液原液罐 Cu/As 的漂移刻意未计算，以保护外部时间留出评价。",
        "",
        "## 2. 数据与字段安全审计",
        "",
        f"- 2024—2025 开发期行数：{development_rows:,}。",
        f"- 2026 外部期行数：{external_rows:,}。",
        f"- 共同冻结过程衍生特征：{len(header_audit.common_process_features):,} 列。",
        f"- 基础过程信号：{base_summary['base_signal'].nunique():,} 个。",
        (
            "- 完整表中的 `origin_cu_g_l`、`origin_as_mg_l` 及其近似/派生命名在数值读取阶段前已排除。"
        ),
        "- `target_`、`next_`、`lead_` 禁列检查：通过（实际载入列中为 0）。",
        (
            f"- 共同特征顺序 SHA256：`{header_audit.common_feature_order_sha256}`。"
        ),
        (
            "- 2026 共同特征顺序与开发期冻结顺序："
            + ("一致。" if header_audit.external_common_order_matches else "不一致。")
        ),
        "",
        "## 3. 漂移指标如何理解",
        "",
        "- 缺失率差 = 2026 缺失率 − 2024—2025 缺失率。",
        (
            "- 稳健中位数位移 =（2026 中位数 − 开发期中位数）/ 开发期 IQR；"
            "开发期 IQR 为 0 时不强行制造有限数值。"
        ),
        "- SMD 使用两个时期非缺失值的合并标准差，仅作描述性尺度统一。",
        (
            "- PSI 的分箱边界只由 2024—2025 开发期十分位数拟合，2026 只被投影到已冻结边界。"
            "本报告分别给出“含缺失箱 PSI”和“条件于非缺失值的 PSI”；前者可由数据可用性变化驱动，"
            "后者才用于描述已观测数值分布变化。0.10/0.25 仅作筛查阈值，不是显著性检验。"
        ),
        "- 超范围比例是在 2026 非缺失值中，低于开发期最小值或高于开发期最大值的比例。",
        "",
        "## 4. 702 个衍生列的描述性概览",
        "",
        (
            f"- 含缺失箱 PSI ≥ 0.10：{int((psi_including_missing >= 0.10).sum()):,} 列；"
            f"≥ 0.25：{int((psi_including_missing >= 0.25).sum()):,} 列。"
        ),
        (
            f"- 非缺失数值特征的条件 PSI ≥ 0.10：{int((numeric_psi >= 0.10).sum()):,} 列；"
            f"≥ 0.25：{int((numeric_psi >= 0.25).sum()):,} 列。"
        ),
        f"- 非缺失数值特征 |SMD| ≥ 0.25：{int((numeric_abs_smd >= 0.25).sum()):,} 列。",
        (
            f"- 非缺失数值的 2026 超出开发期范围比例中位数：{_fmt(numeric_outside.median(), 4)}；"
            f"90 分位数：{_fmt(numeric_outside.quantile(0.90), 4)}。"
        ),
        (
            "这些列由同一基础信号的多个锚点、缺失标志、观测年龄和窗口统计派生，彼此高度相关。"
            "列数只能用于定位，不应被解释为独立证据票数。"
        ),
        "",
        "## 5. 按 27 个基础信号拆分两类漂移",
        "",
        "### 5.1 测量可用性漂移",
        "",
        (
            "这里看 t0/12h 均值是否缺失、七个锚点的缺失率差，以及缺失标志、观测年龄和有效记录数。"
            "它反映数据采集/对齐可用性，不等同于电流、流量、温度等物理量本身发生变化。"
        ),
        "",
        "| 基础信号 | 中文含义 | t0开发缺失率 | t0外部缺失率 | t0缺失率差 | mean12缺失率差 | 锚点缺失率差绝对值中位数 | 可用性特征PSI中位数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ranked_availability.iterrows():
        lines.append(
            "| {base} | {cn} | {dev_miss} | {ext_miss} | {t0_diff} | {mean_diff} | {anchor_diff} | {avail_psi} |".format(
                base=row["base_signal"],
                cn=row["基础信号中文名"] or "—",
                dev_miss=_fmt(row["t0_development_missing_rate"], 4),
                ext_miss=_fmt(row["t0_external_missing_rate"], 4),
                t0_diff=_fmt(
                    row["t0_missing_rate_difference_external_minus_development"], 4
                ),
                mean_diff=_fmt(
                    row["mean12_missing_rate_difference_external_minus_development"], 4
                ),
                anchor_diff=_fmt(row["median_abs_anchor_missing_rate_difference"], 4),
                avail_psi=_fmt(row["median_availability_feature_psi"]),
            )
        )
    lines.extend(
        [
            "",
            "### 5.2 条件于非缺失值的数值漂移",
            "",
            (
                "这里排除缺失箱，只比较已经观测到的数值。SMD 同样只使用两期非缺失值。"
                "该表更接近物理状态差异，但仍是关联性描述，且可能受记录口径和仪表变化影响。"
            ),
            "",
            "| 基础信号 | 中文含义 | t0非缺失PSI | t0 SMD | mean12非缺失PSI | mean12 SMD | 数值特征PSI中位数 | 数值特征SMD绝对值中位数 | t0超范围率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in ranked_numeric.iterrows():
        lines.append(
            "| {base} | {cn} | {t0_psi} | {t0_smd} | {mean_psi} | {mean_smd} | {psi_med} | {smd_med} | {outside} |".format(
                base=row["base_signal"],
                cn=row["基础信号中文名"] or "—",
                t0_psi=_fmt(row["t0_nonmissing_psi"]),
                t0_smd=_fmt(row["t0_nonmissing_smd"]),
                mean_psi=_fmt(row["mean12_nonmissing_psi"]),
                mean_smd=_fmt(row["mean12_nonmissing_smd"]),
                psi_med=_fmt(row["median_numeric_nonmissing_psi"]),
                smd_med=_fmt(row["median_abs_numeric_smd"]),
                outside=_fmt(
                    row["t0_external_outside_development_range_rate"], 4
                ),
            )
        )
    lines.extend(["", "## 6. 域可分性诊断", ""])
    if domain_enabled and not domain_summary.empty:
        lines.extend(
            [
                (
                    "域分类器只用 X 区分“开发期/2026”，同时采用确定性随机分层折和配对连续时间块留出。"
                    "每一折的中位数插补、稳健缩放和分类器参数只在该折训练部分拟合。"
                ),
                "",
                "| 诊断器 | 验证策略 | 折数 | ROC AUC均值±SD | 平衡准确率均值±SD |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for _, row in domain_summary.iterrows():
            lines.append(
                "| {model} | {strategy} | {folds} | {auc} ± {auc_sd} | {ba} ± {ba_sd} |".format(
                    model=row["model_name"],
                    strategy=row["cv_strategy"],
                    folds=int(row["fold_count"]),
                    auc=_fmt(row["roc_auc_mean"]),
                    auc_sd=_fmt(row["roc_auc_std_ddof1"]),
                    ba=_fmt(row["balanced_accuracy_mean"]),
                    ba_sd=_fmt(row["balanced_accuracy_std_ddof1"]),
                )
            )
        lines.extend(
            [
                "",
                (
                    "ROC AUC 越接近 0.5，表示在当前诊断设置下越难区分；越接近 1，表示 X 的时期可分性越强。"
                    "它不是 Cu/As 预测模型的 AUC，也不代表 2026 的 Cu/As 预测误差。"
                ),
                (
                    "报告同时给出随机分层折与“开发期/外部期分别按时间排序、配对留出连续块、"
                    "并从训练集中清除验证块边界前后 24 小时记录”的敏感性结果。"
                    "连续块结果用于检查随机折近邻泄漏，但工业记录仍非独立同分布，二者都不是统计显著性或部署保证。"
                ),
            ]
        )
    else:
        lines.append("本次运行关闭了域分类器；单变量与基础信号聚合审计仍完整执行。")
    lines.extend(
        [
            "",
            "## 7. 综合判断",
            "",
            (
                "仅从过程输入 X 看，当前证据不支持把 2026 与 2024—2025 简单视为“差异很小、"
                "可交换抽样”的同一分布。差异同时包含测量可用性变化和已观测数值变化。"
            ),
            (
                "最突出的可用性变化是 `{base}`（{cn}）：t0 缺失率从 {dev} 变为 {ext}。"
                "这首先说明数据覆盖/对齐口径变化，不能直接表述为物理流量变化。"
            ).format(
                base=top_availability["base_signal"],
                cn=top_availability["基础信号中文名"] or "未命名信号",
                dev=_fmt(top_availability["t0_development_missing_rate"], 4),
                ext=_fmt(top_availability["t0_external_missing_rate"], 4),
            ),
            (
                "条件于非缺失值后，数值漂移排序靠前的基础信号包括："
                + "、".join(
                    f"`{row['base_signal']}`（条件 PSI 中位数 {_fmt(row['median_numeric_nonmissing_psi'])}）"
                    for _, row in top_numeric.iterrows()
                )
                + "。这说明高漂移并非全部由缺失箱造成，但仍可能混有仪表、记录口径和工艺调度差异。"
            ),
        ]
    )
    hgb_block = domain_summary.loc[
        (domain_summary.get("model_name") == "HistGradientBoosting")
        & (
            domain_summary.get("cv_strategy")
            == "PAIRED_PURGED_CONTIGUOUS_TIME_BLOCK"
        )
    ] if not domain_summary.empty else pd.DataFrame()
    if not hgb_block.empty:
        lines.append(
            "非线性域诊断在带 24 小时净空区的连续时间块留出中 ROC AUC 为 "
            f"{_fmt(hgb_block.iloc[0]['roc_auc_mean'])} ± "
            f"{_fmt(hgb_block.iloc[0]['roc_auc_std_ddof1'])}；"
            "可分性在削弱随机折近邻泄漏后仍明显，但这仍不是 Cu/As 预测误差。"
        )
    lines.extend(
        [
            (
                "因此，2026 仍适合作为有意更严格的外部时间留出，用来检验跨工况迁移；"
                "但论文必须把该协变量漂移作为适用域限制报告，不能把 2026 当作与开发期同分布的普通随机测试集。"
            ),
            "",
            "## 8. 对后续研究的约束",
            "",
            "1. 本报告可以用于描述适用域差异和预先声明分层评价，但不能查看 2026 结果后回调 A4。",
            "2. A4、智能体图、阈值和评价协议仍应先在 2024—2025 内冻结，再执行一次性外部评价。",
            "3. 若某基础信号漂移明显，论文中应报告该适用域变化，并对 2026 结果采用谨慎外推表述。",
            "4. 任何 2026 Cu/As 当前值或目标值分析必须经过独立的外部评价释放门，不能由本脚本触发。",
            "",
            "## 9. 可审计工件",
            "",
            "- `feature_shift_metrics_v1.csv`：702 个过程衍生列的缺失率、含缺失箱 PSI 与非缺失数值 PSI/SMD。",
            "- `base_signal_shift_summary_v1.csv`：按 27 个基础信号拆分的可用性漂移与非缺失数值漂移。",
            "- `psi_bins_development_fitted_v1.json`：仅由开发期拟合的 PSI 切点、非缺失计数和含缺失箱计数。",
            "- `feature_contract_audit_v1.json`：字段集合、顺序和哈希审计。",
            "- `domain_classifier_fold_metrics_v1.csv`：随机分层与带净空区连续时间块的域分类逐折诊断。",
            "- `domain_classifier_summary_v1.csv`：域分类折间汇总。",
            "- `run_manifest_v1.json`：输入安全边界、运行版本和输出 SHA256。",
            "",
        ]
    )
    return "\n".join(lines)


def run_covariate_shift_audit(
    development_path: Path,
    external_safe_path: Path,
    output_dir: Path,
    *,
    base_signal_chinese_names: Mapping[str, str] | None = None,
    expected_feature_count: int | None = None,
    expected_base_signal_count: int | None = None,
    run_domain_classifier: bool = True,
    include_hist_gradient_boosting: bool = False,
    domain_cv_splits: int = 5,
    domain_cv_strategies: Sequence[str] = (
        "random_stratified",
        "paired_purged_time_blocks",
    ),
    domain_purge_gap_hours: float = 24.0,
    random_state: int = 20260826,
    safe_view_metadata: Mapping[str, Any] | None = None,
) -> CovariateShiftResult:
    """运行不含标签和当前 Cu/As 的协变量漂移审计并写出可追溯工件。"""

    started = time.perf_counter()
    development_path = Path(development_path).resolve()
    external_safe_path = Path(external_safe_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    header_audit = audit_headers(development_path, external_safe_path)
    common_features = header_audit.common_process_features
    if expected_feature_count is not None and len(common_features) != expected_feature_count:
        raise ValueError(
            f"共同冻结过程特征数应为 {expected_feature_count}，实际为 {len(common_features)}。"
        )
    if header_audit.development_only_process_features or header_audit.external_only_process_features:
        raise ValueError(
            "两期过程特征集合不完全一致；为避免悄悄改变冻结输入，已拒绝继续。"
        )
    if not header_audit.external_common_order_matches:
        raise ValueError("2026 安全视图的过程特征顺序与开发期冻结顺序不一致。")

    development = _load_numeric_features(development_path, common_features)
    external = _load_numeric_features(external_safe_path, common_features)
    feature_metrics, psi_details = calculate_feature_shift_metrics(
        development,
        external,
        base_signal_chinese_names=base_signal_chinese_names,
    )
    base_summary = aggregate_by_base_signal(feature_metrics)
    if expected_base_signal_count is not None and len(base_summary) != expected_base_signal_count:
        raise ValueError(
            f"基础信号数应为 {expected_base_signal_count}，实际为 {len(base_summary)}。"
        )

    if run_domain_classifier:
        development_time = _load_identity_time(development_path)
        external_time = _load_identity_time(external_safe_path)
        models = ["LogisticRegression"]
        if include_hist_gradient_boosting:
            models.append("HistGradientBoosting")
        domain_fold_metrics, domain_summary = run_domain_separability_diagnostic(
            development,
            external,
            development_time=development_time,
            external_time=external_time,
            model_names=models,
            n_splits=domain_cv_splits,
            random_state=random_state,
            cv_strategies=domain_cv_strategies,
            purge_gap_hours=domain_purge_gap_hours,
        )
    else:
        domain_fold_metrics = pd.DataFrame(
            columns=[
                "run_id",
                "model_name",
                "cv_strategy",
                "fold_id",
                "roc_auc",
                "balanced_accuracy",
                "accuracy",
                "preprocessing_fit_scope",
            ]
        )
        domain_summary = pd.DataFrame(
            columns=[
                "run_id",
                "model_name",
                "cv_strategy",
                "fold_count",
                "roc_auc_mean",
                "roc_auc_std_ddof1",
                "balanced_accuracy_mean",
                "balanced_accuracy_std_ddof1",
                "diagnostic_only",
                "not_a4_prediction_model",
            ]
        )

    artifact_paths = {
        "feature_metrics": output_dir / "feature_shift_metrics_v1.csv",
        "base_summary": output_dir / "base_signal_shift_summary_v1.csv",
        "psi_bins": output_dir / "psi_bins_development_fitted_v1.json",
        "contract_audit": output_dir / "feature_contract_audit_v1.json",
        "domain_fold_metrics": output_dir / "domain_classifier_fold_metrics_v1.csv",
        "domain_summary": output_dir / "domain_classifier_summary_v1.csv",
        "human_report": output_dir / "协变量漂移审计报告_V1.md",
    }
    feature_metrics.to_csv(
        artifact_paths["feature_metrics"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    base_summary.to_csv(
        artifact_paths["base_summary"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    domain_fold_metrics.to_csv(
        artifact_paths["domain_fold_metrics"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    domain_summary.to_csv(
        artifact_paths["domain_summary"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    _write_json(
        artifact_paths["psi_bins"],
        {
            "run_id": RUN_ID,
            "fit_scope": "DEVELOPMENT_2024_2025_ONLY",
            "external_values_used_to_fit_bins": False,
            "features": psi_details,
        },
    )
    contract_payload = {
        "run_id": RUN_ID,
        "development_header_sha256": header_audit.development_header_sha256,
        "external_safe_header_sha256": header_audit.external_header_sha256,
        "common_feature_order_sha256": header_audit.common_feature_order_sha256,
        "development_process_feature_count": len(
            header_audit.development_process_features
        ),
        "external_process_feature_count": len(header_audit.external_process_features),
        "common_process_feature_count": len(common_features),
        "development_only_process_features": list(
            header_audit.development_only_process_features
        ),
        "external_only_process_features": list(
            header_audit.external_only_process_features
        ),
        "protected_columns_excluded_before_value_read": list(
            header_audit.protected_columns_excluded
        ),
        "forbidden_columns_loaded": [],
        "development_order_matches": header_audit.development_common_order_matches,
        "external_order_matches": header_audit.external_common_order_matches,
        "common_feature_order": list(common_features),
    }
    _write_json(artifact_paths["contract_audit"], contract_payload)
    artifact_paths["human_report"].write_text(
        _render_report(
            header_audit=header_audit,
            development_rows=len(development),
            external_rows=len(external),
            feature_metrics=feature_metrics,
            base_summary=base_summary,
            domain_summary=domain_summary,
            domain_enabled=run_domain_classifier,
        ),
        encoding="utf-8",
    )

    output_hashes = {
        path.name: sha256_file(path) for path in artifact_paths.values()
    }
    safe_header = read_csv_header(external_safe_path)
    forbidden_in_safe = [
        column
        for column in safe_header
        if is_forbidden_future_or_target_column(column)
        or is_protected_current_result_column(column)
        or column == QUALITY_COLUMN
    ]
    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "inputs": {
            "development_path": str(development_path),
            "external_safe_view_path": str(external_safe_path),
            "development_full_file_sha256": None,
            "external_full_file_sha256": None,
            "full_file_hash_omission_reason": (
                "不扫描含保护列的完整文件正文；仅对显式 usecols 后的安全矩阵形成内容指纹。"
            ),
            "development_selected_process_matrix_sha256": _selected_frame_sha256(
                development
            ),
            "external_selected_process_matrix_sha256": _selected_frame_sha256(external),
            "feature_order_sha256": header_audit.common_feature_order_sha256,
            "safe_view_metadata": dict(safe_view_metadata or {}),
        },
        "counts": {
            "development_rows": len(development),
            "external_rows": len(external),
            "process_derived_features": len(common_features),
            "base_process_signals": len(base_summary),
        },
        "methods": {
            "psi_bins_fit_scope": "DEVELOPMENT_2024_2025_ONLY",
            "psi_requested_quantile_bins": 10,
            "psi_missing_bin": True,
            "shift_decomposition": {
                "measurement_availability": [
                    "missing_indicator",
                    "observation_age",
                    "window_valid_count",
                    "anchor_missing_rate",
                ],
                "nonmissing_numeric_state": [
                    "anchor_value",
                    "window_statistic",
                    "window_slope",
                ],
                "conditional_numeric_psi_excludes_missing_bin": True,
            },
            "domain_classifier_enabled": run_domain_classifier,
            "domain_classifier_models": domain_summary["model_name"]
            .drop_duplicates()
            .tolist()
            if not domain_summary.empty
            else [],
            "domain_cv": (
                list(domain_cv_strategies)
            ),
            "domain_time_block_purge_gap_hours": domain_purge_gap_hours,
            "domain_cv_strategies_reported": domain_summary["cv_strategy"]
            .drop_duplicates()
            .tolist()
            if not domain_summary.empty
            else [],
            "domain_preprocessing_fit_scope": "DOMAIN_TRAIN_FOLD_ONLY",
            "domain_classifier_interpretation": (
                "X 时期可分性诊断；不是 Cu/As 预测器，不表示外部预测误差。"
            ),
            "correlated_feature_control": "702 DERIVED COLUMNS AGGREGATED TO 27 BASE SIGNALS",
        },
        "safety": {
            "development_outcome_ledger_read": False,
            "external_outcome_ledger_read": False,
            "origin_known_results_read": False,
            "target_related_feature_values_read": False,
            "target_related_current_value_drift_computed": False,
            "external_target_values_output": False,
            "input_values_loaded_via_explicit_usecols": True,
            "safe_view_forbidden_columns": forbidden_in_safe,
            "label_columns_in_x": [],
            "a4_model_selection_performed": False,
            "a4_modified_from_shift_results": False,
            "llm_calls_made": 0,
            "network_calls_made": 0,
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "output_sha256": output_hashes,
    }
    if forbidden_in_safe:
        raise SecurityBoundaryError(f"安全视图含禁列：{forbidden_in_safe}")
    manifest_path = output_dir / "run_manifest_v1.json"
    _write_json(manifest_path, manifest)
    return CovariateShiftResult(
        output_dir=output_dir,
        feature_metrics=feature_metrics,
        base_signal_summary=base_summary,
        domain_fold_metrics=domain_fold_metrics,
        domain_summary=domain_summary,
        manifest=manifest,
    )
