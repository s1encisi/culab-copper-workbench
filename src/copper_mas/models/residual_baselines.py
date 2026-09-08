"""P2.1 V1：不改写 P2 V1 的保守残差/变化量基线实验。

所有候选都预测 ``delta = 下一次结果 - 当前已知结果``，随后加回当前结果。
外层评价严格复用 P1 冻结的五折扩展窗口；任何统计量、分组中位数、插补、
标准化与 Ridge 参数都只在对应外折 TRAIN 行上拟合。
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from copper_mas.agents.mode import ModeRuleConfig, infer_process_mode
from copper_mas.models.baselines import (
    DEVELOPMENT_YEARS,
    EXPECTED_FOLDS,
    INPUT_FILENAMES,
    PERSISTENCE_COLUMNS,
    TARGET_COLUMNS,
    PreparedP2Data,
    prepare_p2_data,
    sha256_file,
)


RUN_ID = "P2_1_RESIDUAL_BASELINES_V1_2024_2025"
MODEL_NAMES = (
    "Persistence",
    "TrainMedianDelta",
    "TrainMeanDelta",
    "ModeMedianDelta",
    "ReducedDeltaRidge",
)
DEFAULT_MODE_MIN_GROUP_SIZE = 30
DEFAULT_RIDGE_ALPHA = 1000.0
REDUCED_SUFFIXES = ("__mean_12h", "__slope_per_h_12h", "__valid_count_12h")
CURRENT_RESULT_FEATURES = ("origin_cu_g_l", "origin_as_mg_l")
DEFAULT_SELECTION_PROTOCOL = "contracts/frozen/model_selection_protocol_v1.yaml"


@dataclass(frozen=True)
class PreparedResidualData:
    base: PreparedP2Data
    mode_codes: pd.Series
    reduced_features: pd.DataFrame
    reduced_feature_columns: tuple[str, ...]
    base_signal_names: tuple[str, ...]


@dataclass(frozen=True)
class ResidualRunResult:
    output_dir: Path
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame
    mode_coverage: pd.DataFrame
    manifest: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fold_key(value: str) -> int:
    try:
        return int(str(value).rsplit("_", 1)[-1])
    except ValueError:
        return 999999


def select_reduced_delta_features(
    feature_columns: Iterable[str],
    *,
    expected_signal_count: int = 27,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """冻结为当前 Cu/As 加 27 信号的 mean/slope/valid_count。"""

    ordered = tuple(str(column) for column in feature_columns)
    missing_current = sorted(set(CURRENT_RESULT_FEATURES) - set(ordered))
    if missing_current:
        raise ValueError(f"ReducedDeltaRidge 缺少当前结果: {missing_current}")

    by_signal: dict[str, set[str]] = {}
    for column in ordered:
        for suffix in REDUCED_SUFFIXES:
            if column.endswith(suffix):
                signal = column.removesuffix(suffix)
                by_signal.setdefault(signal, set()).add(suffix)
                break
    incomplete = {
        signal: sorted(set(REDUCED_SUFFIXES) - suffixes)
        for signal, suffixes in by_signal.items()
        if suffixes != set(REDUCED_SUFFIXES)
    }
    if incomplete:
        raise ValueError(f"基础信号缺少约定的 12h 统计列: {incomplete}")
    base_signals = tuple(
        dict.fromkeys(
            column.removesuffix(suffix)
            for column in ordered
            for suffix in REDUCED_SUFFIXES
            if column.endswith(suffix)
        )
    )
    if len(base_signals) != expected_signal_count:
        raise ValueError(
            f"ReducedDeltaRidge 必须覆盖 {expected_signal_count} 个基础信号；实际 {len(base_signals)}"
        )
    selected = tuple(
        column
        for column in ordered
        if column in CURRENT_RESULT_FEATURES or column.endswith(REDUCED_SUFFIXES)
    )
    expected_count = len(CURRENT_RESULT_FEATURES) + expected_signal_count * len(REDUCED_SUFFIXES)
    if len(selected) != expected_count:
        raise ValueError(f"ReducedDeltaRidge 应有 {expected_count} 列；实际 {len(selected)}")
    return selected, base_signals


def build_reduced_delta_ridge(*, alpha: float = DEFAULT_RIDGE_ALPHA) -> Pipeline:
    if alpha <= 0:
        raise ValueError("Ridge alpha 必须大于 0")
    return Pipeline(
        steps=[
            (
                "simple_imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("standard_scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(alpha))),
        ]
    )


def mode_median_delta_predictions(
    train_modes: pd.Series,
    train_delta: pd.Series,
    validation_modes: pd.Series,
    *,
    min_group_size: int = DEFAULT_MODE_MIN_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]:
    """仅由训练折估计 mode 中位数；小组和新 mode 回退全局中位数。"""

    if min_group_size < 2:
        raise ValueError("min_group_size 必须至少为 2")
    if len(train_modes) != len(train_delta):
        raise ValueError("train_modes 与 train_delta 长度不一致")
    train = pd.DataFrame(
        {
            "mode_code": train_modes.astype(str).to_numpy(),
            "delta": pd.to_numeric(train_delta, errors="coerce").to_numpy(),
        }
    )
    if train["delta"].isna().any() or train.empty:
        raise ValueError("训练变化量为空或含缺失")
    stats = (
        train.groupby("mode_code", sort=True)["delta"]
        .agg(train_group_count="size", train_group_median_delta="median")
        .reset_index()
    )
    global_median = float(train["delta"].median())
    count_map = stats.set_index("mode_code")["train_group_count"].to_dict()
    median_map = stats.set_index("mode_code")["train_group_median_delta"].to_dict()
    validation = validation_modes.astype(str)
    counts = validation.map(count_map).fillna(0).astype(int).to_numpy()
    fallback = counts < int(min_group_size)
    predictions = np.array(
        [
            global_median if use_fallback else float(median_map[mode])
            for mode, use_fallback in zip(validation, fallback, strict=True)
        ],
        dtype=float,
    )
    return predictions, counts, fallback.astype(bool), global_median, stats


def prepare_residual_data(input_dir: Path) -> PreparedResidualData:
    base = prepare_p2_data(input_dir)
    selected, base_signals = select_reduced_delta_features(base.feature_columns)
    reduced = base.features.loc[:, list(selected)].copy()
    modes: list[str] = []
    for index, record in base.records.iterrows():
        card = infer_process_mode(
            origin_event_id=str(record["origin_event_id"]),
            decision_at=pd.Timestamp(record["decision_at"]).to_pydatetime(),
            feature_row=base.features.iloc[index].to_dict(),
        )
        modes.append(card.mode_code)
    mode_codes = pd.Series(modes, index=base.records.index, name="mode_code", dtype="string")
    if mode_codes.isna().any() or mode_codes.str.strip().eq("").any():
        raise ValueError("A2 mode_code 不能为空")
    return PreparedResidualData(
        base=base,
        mode_codes=mode_codes,
        reduced_features=reduced,
        reduced_feature_columns=selected,
        base_signal_names=base_signals,
    )


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actual_delta: np.ndarray,
    predicted_delta: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "delta_r2": float(r2_score(actual_delta, predicted_delta)),
        "delta_bias": float(np.mean(predicted_delta - actual_delta)),
    }


def _improvement(reference: float, candidate: float) -> float:
    if reference == 0:
        return 0.0 if candidate == 0 else float("nan")
    return float((reference - candidate) / reference * 100.0)


def _delta_summary_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    persistence = predictions.loc[predictions["model_name"] == "Persistence"].copy()
    rows: list[dict[str, Any]] = []
    for (fold_id, target_name), group in persistence.groupby(
        ["fold_id", "target_name"], sort=True
    ):
        rows.append(_one_delta_summary(group, scope="FOLD", fold_id=fold_id, target_name=target_name))
    for target_name, group in persistence.groupby("target_name", sort=True):
        rows.append(_one_delta_summary(group, scope="POOLED", fold_id="POOLED", target_name=target_name))
    return rows


def _one_delta_summary(
    group: pd.DataFrame,
    *,
    scope: str,
    fold_id: str,
    target_name: str,
) -> dict[str, Any]:
    delta = group["actual_delta"].to_numpy(dtype=float)
    return {
        "scope": scope,
        "fold_id": fold_id,
        "target_name": target_name,
        "sample_count": len(delta),
        "mean_delta": np.mean(delta),
        "median_delta": np.median(delta),
        "std_delta": np.std(delta, ddof=1),
        "mean_absolute_delta": np.mean(np.abs(delta)),
        "root_mean_square_delta": np.sqrt(np.mean(np.square(delta))),
        "q05_delta": np.quantile(delta, 0.05),
        "q25_delta": np.quantile(delta, 0.25),
        "q75_delta": np.quantile(delta, 0.75),
        "q95_delta": np.quantile(delta, 0.95),
        "positive_rate": np.mean(delta > 0),
        "negative_rate": np.mean(delta < 0),
        "zero_rate": np.mean(delta == 0),
    }


def _build_mode_coverage(
    predictions: pd.DataFrame,
    *,
    min_group_size: int,
) -> pd.DataFrame:
    mode_rows = predictions.loc[predictions["model_name"] == "ModeMedianDelta"].copy()
    rows: list[dict[str, Any]] = []
    for (fold_id, target_name, mode_code), group in mode_rows.groupby(
        ["fold_id", "target_name", "mode_code"], sort=True
    ):
        fallback_count = int(group["mode_fallback"].sum())
        count = len(group)
        rows.append(
            {
                "scope": "FOLD",
                "fold_id": fold_id,
                "target_name": target_name,
                "mode_code": mode_code,
                "min_group_size": min_group_size,
                "train_group_count_min": int(group["mode_train_group_count"].min()),
                "train_group_count_max": int(group["mode_train_group_count"].max()),
                "validation_count": count,
                "mode_specific_count": count - fallback_count,
                "fallback_count": fallback_count,
                "mode_specific_rate": (count - fallback_count) / count,
                "fallback_rate": fallback_count / count,
            }
        )
    for (target_name, mode_code), group in mode_rows.groupby(
        ["target_name", "mode_code"], sort=True
    ):
        fallback_count = int(group["mode_fallback"].sum())
        count = len(group)
        rows.append(
            {
                "scope": "POOLED",
                "fold_id": "POOLED",
                "target_name": target_name,
                "mode_code": mode_code,
                "min_group_size": min_group_size,
                "train_group_count_min": int(group["mode_train_group_count"].min()),
                "train_group_count_max": int(group["mode_train_group_count"].max()),
                "validation_count": count,
                "mode_specific_count": count - fallback_count,
                "fallback_count": fallback_count,
                "mode_specific_rate": (count - fallback_count) / count,
                "fallback_rate": fallback_count / count,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scope", "target_name", "fold_id", "mode_code"], kind="mergesort"
    ).reset_index(drop=True)


def _overall_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target_name, model_name), group in predictions.groupby(
        ["target_name", "model_name"], sort=True
    ):
        primary = _metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            group["actual_delta"].to_numpy(),
            group["predicted_delta"].to_numpy(),
        )
        persistence = group.assign(
            y_pred=group["current_value"], predicted_delta=0.0
        )
        reference = _metrics(
            persistence["y_true"].to_numpy(),
            persistence["y_pred"].to_numpy(),
            persistence["actual_delta"].to_numpy(),
            persistence["predicted_delta"].to_numpy(),
        )
        per_fold = fold_metrics.loc[
            (fold_metrics["target_name"] == target_name)
            & (fold_metrics["model_name"] == model_name)
        ]
        row: dict[str, Any] = {
            "run_id": RUN_ID,
            "target_name": target_name,
            "model_name": model_name,
            "validation_sample_count": len(group),
            "fold_count": per_fold["fold_id"].nunique(),
            "pooled_mae": primary["mae"],
            "pooled_rmse": primary["rmse"],
            "pooled_r2": primary["r2"],
            "pooled_delta_r2": primary["delta_r2"],
            "pooled_delta_bias": primary["delta_bias"],
            "pooled_persistence_mae": reference["mae"],
            "pooled_persistence_rmse": reference["rmse"],
            "pooled_persistence_r2": reference["r2"],
            "pooled_relative_mae_improvement_vs_persistence_pct": _improvement(
                reference["mae"], primary["mae"]
            ),
            "pooled_relative_rmse_improvement_vs_persistence_pct": _improvement(
                reference["rmse"], primary["rmse"]
            ),
            "pooled_r2_delta_vs_persistence": primary["r2"] - reference["r2"],
            "fit_time_seconds_total": per_fold["fit_time_seconds"].sum(),
            "predict_time_seconds_total": per_fold["predict_time_seconds"].sum(),
        }
        for metric in (
            "mae",
            "rmse",
            "r2",
            "delta_r2",
            "delta_bias",
            "relative_mae_improvement_vs_persistence_pct",
            "relative_rmse_improvement_vs_persistence_pct",
        ):
            row[f"fold_{metric}_mean"] = per_fold[metric].mean()
            row[f"fold_{metric}_std"] = per_fold[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target_name", "model_name"], kind="mergesort"
    ).reset_index(drop=True)


def load_selection_protocol(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"冻结选模协议不存在: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("冻结选模协议必须是 YAML mapping")
    required = {
        "protocol_id",
        "status",
        "candidate_allowlist",
        "hard_gates",
        "reference_model",
        "promotion_thresholds",
        "complexity_rank",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"冻结选模协议缺少字段: {missing}")
    if payload["status"] != "FROZEN_BEFORE_P2_1_RESULTS":
        raise ValueError("选模协议不是结果生成前冻结状态")
    if payload["reference_model"] != "Persistence":
        raise ValueError("P2.1 只接受 Persistence 作为冻结参考")
    allowlist = set(map(str, payload["candidate_allowlist"]))
    absent = sorted(set(MODEL_NAMES) - allowlist)
    if absent:
        raise ValueError(f"P2.1 模型不在冻结 allowlist: {absent}")
    thresholds = payload["promotion_thresholds"]
    required_thresholds = {
        "pooled_mae_ratio_vs_persistence_max",
        "pooled_rmse_ratio_vs_persistence_max",
        "folds_with_mae_improvement_min",
        "worst_fold_mae_ratio_vs_persistence_max",
    }
    if required_thresholds - set(thresholds):
        raise ValueError("冻结协议的 promotion_thresholds 不完整")
    return payload


def evaluate_protocol_eligibility(
    overall: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    hard_gates_passed: bool,
) -> pd.DataFrame:
    """按冻结阈值作只读门槛判断；不排序、不选择、不写 winner。"""

    thresholds = protocol["promotion_thresholds"]
    complexity = {str(k): int(v) for k, v in protocol["complexity_rank"].items()}
    rows: list[dict[str, Any]] = []
    for target_name in sorted(overall["target_name"].unique()):
        target_overall = overall.loc[overall["target_name"] == target_name]
        reference_rows = target_overall.loc[target_overall["model_name"] == "Persistence"]
        if len(reference_rows) != 1:
            raise ValueError(f"{target_name} 缺少唯一 Persistence 总体指标")
        reference = reference_rows.iloc[0]
        reference_folds = fold_metrics.loc[
            (fold_metrics["target_name"] == target_name)
            & (fold_metrics["model_name"] == "Persistence")
        ].set_index("fold_id")
        if len(reference_folds) != EXPECTED_FOLDS:
            raise ValueError(f"{target_name} Persistence 不是冻结 {EXPECTED_FOLDS} 折")
        for _, candidate in target_overall.iterrows():
            model_name = str(candidate["model_name"])
            candidate_folds = fold_metrics.loc[
                (fold_metrics["target_name"] == target_name)
                & (fold_metrics["model_name"] == model_name)
            ].set_index("fold_id")
            if set(candidate_folds.index) != set(reference_folds.index):
                raise ValueError(f"{target_name}/{model_name} 折集合与 Persistence 不一致")
            fold_ratios = (
                candidate_folds.loc[reference_folds.index, "mae"].astype(float)
                / reference_folds["mae"].astype(float)
            )
            pooled_mae_ratio = float(candidate["pooled_mae"] / reference["pooled_mae"])
            pooled_rmse_ratio = float(candidate["pooled_rmse"] / reference["pooled_rmse"])
            folds_improved = int((fold_ratios < 1.0).sum())
            worst_fold_ratio = float(fold_ratios.max())
            gates = {
                "gate_pooled_mae": pooled_mae_ratio
                <= float(thresholds["pooled_mae_ratio_vs_persistence_max"]),
                "gate_pooled_rmse": pooled_rmse_ratio
                <= float(thresholds["pooled_rmse_ratio_vs_persistence_max"]),
                "gate_fold_wins": folds_improved
                >= int(thresholds["folds_with_mae_improvement_min"]),
                "gate_worst_fold": worst_fold_ratio
                <= float(thresholds["worst_fold_mae_ratio_vs_persistence_max"]),
            }
            is_reference = model_name == "Persistence"
            eligible = bool(hard_gates_passed and all(gates.values()) and not is_reference)
            if is_reference:
                status = "REFERENCE_MODEL_NOT_ASSESSED_FOR_PROMOTION"
            elif eligible:
                status = "PASSES_FROZEN_GATES_READ_ONLY"
            else:
                status = "FAILS_FROZEN_GATES_READ_ONLY"
            rows.append(
                {
                    "protocol_id": protocol["protocol_id"],
                    "assessment_mode": "READ_ONLY_NO_WINNER_WRITE",
                    "target_name": target_name,
                    "model_name": model_name,
                    "is_reference_model": is_reference,
                    "hard_gates_passed": bool(hard_gates_passed),
                    "pooled_mae_ratio_vs_persistence": pooled_mae_ratio,
                    "pooled_mae_ratio_max": float(
                        thresholds["pooled_mae_ratio_vs_persistence_max"]
                    ),
                    "gate_pooled_mae_passed": gates["gate_pooled_mae"],
                    "pooled_rmse_ratio_vs_persistence": pooled_rmse_ratio,
                    "pooled_rmse_ratio_max": float(
                        thresholds["pooled_rmse_ratio_vs_persistence_max"]
                    ),
                    "gate_pooled_rmse_passed": gates["gate_pooled_rmse"],
                    "folds_with_mae_improvement": folds_improved,
                    "folds_with_mae_improvement_min": int(
                        thresholds["folds_with_mae_improvement_min"]
                    ),
                    "gate_fold_wins_passed": gates["gate_fold_wins"],
                    "worst_fold_mae_ratio_vs_persistence": worst_fold_ratio,
                    "worst_fold_mae_ratio_max": float(
                        thresholds["worst_fold_mae_ratio_vs_persistence_max"]
                    ),
                    "gate_worst_fold_passed": gates["gate_worst_fold"],
                    "median_fold_mae_ratio_vs_persistence": float(fold_ratios.median()),
                    "fold_mae_ratio_standard_deviation": float(fold_ratios.std(ddof=1)),
                    "complexity_rank": complexity.get(model_name),
                    "eligible_for_promotion": eligible,
                    "eligibility_status": status,
                    "selected_model_written": False,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target_name", "model_name"], kind="mergesort"
    ).reset_index(drop=True)


def _fmt(value: Any, digits: int = 4) -> str:
    number = float(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _render_report(
    data: PreparedResidualData,
    fold_metrics: pd.DataFrame,
    overall: pd.DataFrame,
    delta_summary: pd.DataFrame,
    mode_coverage: pd.DataFrame,
    eligibility: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    min_group_size: int,
    ridge_alpha: float,
) -> str:
    pooled_delta = delta_summary.loc[delta_summary["scope"] == "POOLED"]
    pooled_modes = mode_coverage.loc[
        (mode_coverage["scope"] == "POOLED")
        & (mode_coverage["target_name"] == "target_cu_g_l")
    ].sort_values("validation_count", ascending=False)
    pooled_mode_n = int(pooled_modes["validation_count"].sum())
    pooled_mode_fallback = int(pooled_modes["fallback_count"].sum())
    pooled_mode_fallback_rate = (
        pooled_mode_fallback / pooled_mode_n if pooled_mode_n else float("nan")
    )
    eligible_count = int(eligibility["eligible_for_promotion"].sum())
    lines = [
        "# P2.1 保守残差/变化量基线实验说明（V1）",
        "",
        "## 1. 实验目的与边界",
        "",
        (
            "本实验保持 P2 V1 的 2,953 条 `tier1_core_primary_prospective` 主样本与"
            "冻结五折不变，只把学习目标改为 `下一次结果 − 当前结果`。共有 2,732 条起点进入"
            "一次且仅一次的 OOF 验证。未读取 2026、未随机拆分、未调用 LLM，也未改写 P2 V1 工件。"
        ),
        "",
        (
            "全部五个候选均完整保留，不按单折或 pooled 最好结果强选 winner。结果用于检验"
            "Persistence 为什么强，以及有限过程摘要能否稳定解释下一次变化。"
        ),
        "",
        "## 2. 五个候选",
        "",
        "- `Persistence`：预测变化量恒为 0。",
        "- `TrainMedianDelta`：每折仅用 TRAIN 变化量中位数。",
        "- `TrainMeanDelta`：每折仅用 TRAIN 变化量均值。",
        (
            f"- `ModeMedianDelta`：按当前确定性 A2 四回路 `mode_code` 使用 TRAIN 组中位数；"
            f"训练组少于 {min_group_size} 条时回退该折全局中位数。"
        ),
        (
            f"- `ReducedDeltaRidge`：固定 `alpha={ridge_alpha:g}`；只用当前 Cu/As 与 27 个信号"
            "的 12 h mean、slope、valid_count，共 83 列。"
        ),
        "",
        (
            "ReducedDeltaRidge 的 SimpleImputer(add_indicator=True)、StandardScaler 和 Ridge"
            "位于同一 Pipeline，并在每个外折 TRAIN 内重新拟合。没有用外折验证调 alpha。"
        ),
        "",
        "## 3. OOF 总体指标",
        "",
        "| 目标 | 模型 | n | MAE | RMSE | R² | delta R² | MAE 相对 Persistence | RMSE 相对 Persistence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        digits = 3 if row["target_name"] == "target_as_mg_l" else 4
        lines.append(
            "| {target} | {model} | {n} | {mae} | {rmse} | {r2} | {dr2} | {mi}% | {ri}% |".format(
                target=row["target_name"],
                model=row["model_name"],
                n=int(row["validation_sample_count"]),
                mae=_fmt(row["pooled_mae"], digits),
                rmse=_fmt(row["pooled_rmse"], digits),
                r2=_fmt(row["pooled_r2"], 4),
                dr2=_fmt(row["pooled_delta_r2"], 4),
                mi=_fmt(row["pooled_relative_mae_improvement_vs_persistence_pct"], 2),
                ri=_fmt(row["pooled_relative_rmse_improvement_vs_persistence_pct"], 2),
            )
        )
    lines.extend(
        [
            "",
            "Cu 单位为 g/L，As 单位为 mg/L；负的相对改进表示不如 Persistence。",
            "",
            "## 4. 实际变化量分布",
            "",
            "| 目标 | n | mean | median | mean absolute | RMS | q05 | q95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in pooled_delta.iterrows():
        digits = 3 if row["target_name"] == "target_as_mg_l" else 4
        lines.append(
            "| {target} | {n} | {mean} | {median} | {ma} | {rms} | {q05} | {q95} |".format(
                target=row["target_name"],
                n=int(row["sample_count"]),
                mean=_fmt(row["mean_delta"], digits),
                median=_fmt(row["median_delta"], digits),
                ma=_fmt(row["mean_absolute_delta"], digits),
                rms=_fmt(row["root_mean_square_delta"], digits),
                q05=_fmt(row["q05_delta"], digits),
                q95=_fmt(row["q95_delta"], digits),
            )
        )
    delta_by_target = pooled_delta.set_index("target_name")
    cu_delta = delta_by_target.loc["target_cu_g_l"]
    as_delta = delta_by_target.loc["target_as_mg_l"]
    lines.extend(
        [
            "",
            (
                "Cu 的 pooled 平均变化仅为 "
                f"{_fmt(cu_delta['mean_delta'], 4)} g/L（变化 RMS "
                f"{_fmt(cu_delta['root_mean_square_delta'], 4)} g/L）；As 的平均变化仅为 "
                f"{_fmt(as_delta['mean_delta'], 3)} mg/L（变化 RMS "
                f"{_fmt(as_delta['root_mean_square_delta'], 3)} mg/L）。这解释了零变化预测为何很难被"
                "简单常数修正稳定超越。"
            ),
            "",
            (
                "Persistence 的绝对值 R² 可以很高，因为当前水平本身携带强信息；真正检验变化可预测性"
                "应同时看 delta R²、相对 MAE/RMSE 以及五折稳定性。不能只因绝对值 R² 高就推断过程变量"
                "没有作用。"
            ),
            "",
            "## 5. A2 mode 覆盖与回退（POOLED，Cu/As 覆盖相同）",
            "",
            (
                f"固定最小训练组数为 {min_group_size}。共 {pooled_mode_n} 条 OOF 验证中，"
                f"{pooled_mode_fallback} 条回退全局训练中位数，整体回退率 "
                f"{pooled_mode_fallback_rate:.2%}。"
            ),
            "",
            "| mode_code | 验证数 | 使用组中位数 | 回退全局 | 覆盖率 | 回退率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in pooled_modes.iterrows():
        lines.append(
            "| `{mode}` | {n} | {specific} | {fallback} | {rate:.2%} | {fallback_rate:.2%} |".format(
                mode=row["mode_code"],
                n=int(row["validation_count"]),
                specific=int(row["mode_specific_count"]),
                fallback=int(row["fallback_count"]),
                rate=float(row["mode_specific_rate"]),
                fallback_rate=float(row["fallback_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## 6. 冻结选模协议的只读 eligibility 判断",
            "",
            (
                f"协议 `{protocol['protocol_id']}` 在本次结果生成前冻结。晋级要求为：pooled MAE"
                " 相对 Persistence 至少改善 2%，pooled RMSE 不变差，至少 3/5 折 MAE 改善，"
                "且最差折 MAE 比不超过 1.25。下表只判断门槛，不自动选择或写入 winner。"
            ),
            "",
            "| 目标 | 模型 | MAE比 | RMSE比 | 胜出折 | 最差折比 | eligibility |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in eligibility.iterrows():
        lines.append(
            "| {target} | {model} | {mae} | {rmse} | {wins}/5 | {worst} | {status} |".format(
                target=row["target_name"],
                model=row["model_name"],
                mae=_fmt(row["pooled_mae_ratio_vs_persistence"], 4),
                rmse=_fmt(row["pooled_rmse_ratio_vs_persistence"], 4),
                wins=int(row["folds_with_mae_improvement"]),
                worst=_fmt(row["worst_fold_mae_ratio_vs_persistence"], 4),
                status=row["eligibility_status"],
            )
        )
    lines.extend(
        [
            "",
            (
                f"满足全部冻结晋级门槛的非参考候选数为 {eligible_count}。该数字只是只读"
                " eligibility 结果，脚本没有写入最终 winner；若无候选通过，后续应按协议"
                "回退规则另行冻结 Persistence，而不是由本实验自动改写。"
            ),
            "",
            "## 7. 逐折结果与解释边界",
            "",
            "| 目标 | 模型 | 折 | n | MAE | RMSE | R² | delta R² | MAE相对改进 | RMSE相对改进 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in fold_metrics.sort_values(
        ["target_name", "model_name", "fold_id"], kind="mergesort"
    ).iterrows():
        digits = 3 if row["target_name"] == "target_as_mg_l" else 4
        lines.append(
            "| {target} | {model} | {fold} | {n} | {mae} | {rmse} | {r2} | {dr2} | {mi}% | {ri}% |".format(
                target=row["target_name"],
                model=row["model_name"],
                fold=row["fold_id"],
                n=int(row["validation_sample_count"]),
                mae=_fmt(row["mae"], digits),
                rmse=_fmt(row["rmse"], digits),
                r2=_fmt(row["r2"], 4),
                dr2=_fmt(row["delta_r2"], 4),
                mi=_fmt(row["relative_mae_improvement_vs_persistence_pct"], 2),
                ri=_fmt(row["relative_rmse_improvement_vs_persistence_pct"], 2),
            )
        )
    lines.extend(
        [
            "",
            "折间均值与样本标准差完整保存在 `overall_metrics_v1.csv`。",
            "",
            (
                "若某个残差模型只在部分折改善，不能删除不利折或称其稳定优于 Persistence；"
                "若全部模型均未稳定改善，也只能说明当前历史数据、特征摘要与合同下的变化量信号弱或"
                "漂移强，不能外推为所有过程信息或所有模型永久无效。"
            ),
            "",
            "## 8. 可审计文件",
            "",
            "- `oof_residual_predictions_v1.csv`：逐样本实际/预测变化量与加回后的绝对预测。",
            "- `fold_metrics_v1.csv`、`overall_metrics_v1.csv`：逐折、pooled 和折间离散。",
            "- `delta_summary_v1.csv`：实际变化量分布。",
            "- `mode_coverage_v1.csv`：每折及 pooled 的各 mode 覆盖与回退。",
            "- `model_eligibility_v1.csv`：冻结协议的逐目标只读门槛证据。",
            "- `reduced_feature_manifest_v1.csv`：83 个 ReducedDeltaRidge 特征。",
            "- `experiment_config_v1.json`：固定最小组数、Ridge alpha 与无选择规则。",
            "- `run_manifest_v1.json`：输入/输出 SHA256、运行环境与安全声明。",
            "",
        ]
    )
    return "\n".join(lines)


def run_residual_baselines(
    input_dir: Path,
    output_dir: Path,
    *,
    min_mode_group_size: int = DEFAULT_MODE_MIN_GROUP_SIZE,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    selection_protocol_path: Path | None = None,
) -> ResidualRunResult:
    started = time.perf_counter()
    generated_at = datetime.now(timezone.utc)
    if min_mode_group_size < 2:
        raise ValueError("min_mode_group_size 必须至少为 2")
    data = prepare_residual_data(input_dir)
    if selection_protocol_path is None:
        raise ValueError("必须显式提供结果生成前冻结的 selection_protocol_path")
    selection_protocol_path = Path(selection_protocol_path).resolve()
    protocol = load_selection_protocol(selection_protocol_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = data.base.records.copy()
    records["mode_code"] = data.mode_codes
    records_by_pair = records.set_index("pair_id")
    full_features = data.base.features.copy()
    full_features.index = records["pair_id"]
    reduced_features = data.reduced_features.copy()
    reduced_features.index = records["pair_id"]
    targets = data.base.targets.copy()
    targets.index = records["pair_id"]
    mode_by_pair = data.mode_codes.copy()
    mode_by_pair.index = records["pair_id"]

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_ids = sorted(data.base.folds["fold_id"].unique(), key=_fold_key)
    for fold_id in fold_ids:
        fold = data.base.folds.loc[data.base.folds["fold_id"] == fold_id]
        train_ids = fold.loc[fold["fold_role"] == "TRAIN"].sort_values("decision_at")["pair_id"].astype(str)
        validation_ids = fold.loc[fold["fold_role"] == "VALIDATION"].sort_values("decision_at")["pair_id"].astype(str)
        validation_meta = records_by_pair.loc[
            validation_ids, ["origin_event_id", "decision_at", "mode_code"]
        ].reset_index()

        for target_name in TARGET_COLUMNS:
            current_column = PERSISTENCE_COLUMNS[target_name]
            train_current = full_features.loc[train_ids, current_column].to_numpy(dtype=float)
            validation_current = full_features.loc[validation_ids, current_column].to_numpy(dtype=float)
            train_target = targets.loc[train_ids, target_name].to_numpy(dtype=float)
            y_true = targets.loc[validation_ids, target_name].to_numpy(dtype=float)
            train_delta = train_target - train_current
            actual_delta = y_true - validation_current
            persistence_metrics = _metrics(
                y_true,
                validation_current,
                actual_delta,
                np.zeros_like(actual_delta),
            )

            fit_started = time.perf_counter()
            train_median = float(np.median(train_delta))
            median_fit_time = time.perf_counter() - fit_started
            fit_started = time.perf_counter()
            train_mean = float(np.mean(train_delta))
            mean_fit_time = time.perf_counter() - fit_started
            fit_started = time.perf_counter()
            mode_delta, mode_counts, mode_fallback, global_median, _ = mode_median_delta_predictions(
                mode_by_pair.loc[train_ids],
                pd.Series(train_delta, index=train_ids.index),
                mode_by_pair.loc[validation_ids],
                min_group_size=min_mode_group_size,
            )
            mode_fit_time = time.perf_counter() - fit_started
            ridge = build_reduced_delta_ridge(alpha=ridge_alpha)
            fit_started = time.perf_counter()
            ridge.fit(
                reduced_features.loc[train_ids, list(data.reduced_feature_columns)],
                train_delta,
            )
            ridge_fit_time = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            ridge_delta = np.asarray(
                ridge.predict(
                    reduced_features.loc[validation_ids, list(data.reduced_feature_columns)]
                ),
                dtype=float,
            )
            ridge_predict_time = time.perf_counter() - predict_started

            model_deltas: dict[str, tuple[np.ndarray, float, float]] = {
                "Persistence": (np.zeros_like(actual_delta), 0.0, 0.0),
                "TrainMedianDelta": (
                    np.full_like(actual_delta, train_median),
                    median_fit_time,
                    0.0,
                ),
                "TrainMeanDelta": (
                    np.full_like(actual_delta, train_mean),
                    mean_fit_time,
                    0.0,
                ),
                "ModeMedianDelta": (mode_delta, mode_fit_time, 0.0),
                "ReducedDeltaRidge": (ridge_delta, ridge_fit_time, ridge_predict_time),
            }
            for model_name, (predicted_delta, fit_time, predict_time) in model_deltas.items():
                y_pred = validation_current + predicted_delta
                current_metrics = _metrics(y_true, y_pred, actual_delta, predicted_delta)
                metric_rows.append(
                    {
                        "run_id": RUN_ID,
                        "fold_id": fold_id,
                        "target_name": target_name,
                        "model_name": model_name,
                        "train_sample_count": len(train_ids),
                        "validation_sample_count": len(validation_ids),
                        **current_metrics,
                        "persistence_mae": persistence_metrics["mae"],
                        "persistence_rmse": persistence_metrics["rmse"],
                        "persistence_r2": persistence_metrics["r2"],
                        "relative_mae_improvement_vs_persistence_pct": _improvement(
                            persistence_metrics["mae"], current_metrics["mae"]
                        ),
                        "relative_rmse_improvement_vs_persistence_pct": _improvement(
                            persistence_metrics["rmse"], current_metrics["rmse"]
                        ),
                        "r2_delta_vs_persistence": current_metrics["r2"] - persistence_metrics["r2"],
                        "fit_time_seconds": fit_time,
                        "predict_time_seconds": predict_time,
                    }
                )
                is_mode = model_name == "ModeMedianDelta"
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "run_id": RUN_ID,
                            "fold_id": fold_id,
                            "pair_id": validation_meta["pair_id"].astype(str),
                            "origin_event_id": validation_meta["origin_event_id"].astype(str),
                            "decision_at": pd.to_datetime(validation_meta["decision_at"]),
                            "mode_code": validation_meta["mode_code"].astype(str),
                            "target_name": target_name,
                            "model_name": model_name,
                            "current_value": validation_current,
                            "y_true": y_true,
                            "actual_delta": actual_delta,
                            "predicted_delta": predicted_delta,
                            "y_pred": y_pred,
                            "absolute_error": np.abs(y_true - y_pred),
                            "squared_error": np.square(y_true - y_pred),
                            "mode_train_group_count": (
                                mode_counts.astype(float)
                                if is_mode
                                else np.full(len(y_true), np.nan)
                            ),
                            "mode_fallback": (
                                mode_fallback.astype(float)
                                if is_mode
                                else np.full(len(y_true), np.nan)
                            ),
                            "mode_global_train_median_delta": (
                                np.full(len(y_true), global_median)
                                if is_mode
                                else np.full(len(y_true), np.nan)
                            ),
                        }
                    )
                )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["target_name", "model_name", "fold_id", "decision_at", "pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["target_name", "model_name", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    overall = _overall_metrics(predictions, fold_metrics)
    finite_prediction_fields = predictions[
        ["current_value", "y_true", "actual_delta", "predicted_delta", "y_pred"]
    ].to_numpy(dtype=float)
    if not np.isfinite(finite_prediction_fields).all():
        raise ValueError("P2.1 预测工件包含非有限值")
    unique_oof_ok = all(
        not group["pair_id"].duplicated().any()
        for _, group in predictions.groupby(["target_name", "model_name"], sort=False)
    )
    if not unique_oof_ok:
        raise ValueError("同一目标/模型的 OOF pair_id 不唯一")
    hard_gates_passed = bool(
        len(fold_ids) == EXPECTED_FOLDS
        and unique_oof_ok
        and set(data.base.records["decision_at"].dt.year).issubset(DEVELOPMENT_YEARS)
    )
    eligibility = evaluate_protocol_eligibility(
        overall,
        fold_metrics,
        protocol,
        hard_gates_passed=hard_gates_passed,
    )
    delta_summary = pd.DataFrame(_delta_summary_rows(predictions)).sort_values(
        ["scope", "target_name", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    mode_coverage = _build_mode_coverage(
        predictions, min_group_size=min_mode_group_size
    )
    feature_manifest = pd.DataFrame(
        {
            "feature_order": np.arange(len(data.reduced_feature_columns)),
            "feature_name": list(data.reduced_feature_columns),
            "source_artifact": INPUT_FILENAMES["features"],
            "availability": "KNOWN_AT_DECISION",
            "role": [
                "CURRENT_RESULT" if feature in CURRENT_RESULT_FEATURES else "PROCESS_12H_SUMMARY"
                for feature in data.reduced_feature_columns
            ],
        }
    )
    config: dict[str, Any] = {
        "run_id": RUN_ID,
        "sample": "tier1_core_primary_prospective",
        "development_years": [2024, 2025],
        "external_2026_used": False,
        "targets": list(TARGET_COLUMNS),
        "models": list(MODEL_NAMES),
        "mode_median_delta": {
            "mode_source": "A2 deterministic four-circuit mode_code",
            "min_train_group_size": min_mode_group_size,
            "fallback": "outer-train global median delta",
            "rule_config": ModeRuleConfig().__dict__,
        },
        "reduced_delta_ridge": {
            "alpha": ridge_alpha,
            "alpha_selection": "fixed before outer validation; no tuning",
            "feature_count": len(data.reduced_feature_columns),
            "base_signal_count": len(data.base_signal_names),
            "pipeline": [
                "SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)",
                "StandardScaler()",
                f"Ridge(alpha={ridge_alpha!r})",
            ],
        },
        "cross_validation": {
            "source": INPUT_FILENAMES["folds"],
            "fold_count": EXPECTED_FOLDS,
            "random_split": False,
            "preprocessing_fit_scope": "TRAIN_FOLD_ONLY",
        },
        "selection": {
            "performed": False,
            "selected_model": None,
            "statement": "保留所有结果，不以单折或 pooled 最好值强选 winner。",
        },
        "frozen_selection_protocol": {
            "protocol_id": protocol["protocol_id"],
            "status": protocol["status"],
            "path": str(selection_protocol_path),
            "sha256": sha256_file(selection_protocol_path),
            "assessment": "read_only_eligibility_no_winner_write",
            "promotion_thresholds": protocol["promotion_thresholds"],
        },
    }

    paths = {
        "predictions": output_dir / "oof_residual_predictions_v1.csv",
        "fold_metrics": output_dir / "fold_metrics_v1.csv",
        "overall_metrics": output_dir / "overall_metrics_v1.csv",
        "delta_summary": output_dir / "delta_summary_v1.csv",
        "mode_coverage": output_dir / "mode_coverage_v1.csv",
        "eligibility": output_dir / "model_eligibility_v1.csv",
        "feature_manifest": output_dir / "reduced_feature_manifest_v1.csv",
        "config": output_dir / "experiment_config_v1.json",
        "report": output_dir / "P2.1保守残差基线结果说明_V1.md",
    }
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig", float_format="%.10g")
    fold_metrics.to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig", float_format="%.10g")
    overall.to_csv(paths["overall_metrics"], index=False, encoding="utf-8-sig", float_format="%.10g")
    delta_summary.to_csv(paths["delta_summary"], index=False, encoding="utf-8-sig", float_format="%.10g")
    mode_coverage.to_csv(paths["mode_coverage"], index=False, encoding="utf-8-sig", float_format="%.10g")
    eligibility.to_csv(paths["eligibility"], index=False, encoding="utf-8-sig", float_format="%.10g")
    feature_manifest.to_csv(paths["feature_manifest"], index=False, encoding="utf-8-sig")
    _write_json(paths["config"], config)
    paths["report"].write_text(
        _render_report(
            data,
            fold_metrics,
            overall,
            delta_summary,
            mode_coverage,
            eligibility,
            protocol,
            min_group_size=min_mode_group_size,
            ridge_alpha=ridge_alpha,
        ),
        encoding="utf-8",
    )

    source_hashes = {
        INPUT_FILENAMES[key]: sha256_file(path) for key, path in data.base.source_paths.items()
    }
    source_hashes[selection_protocol_path.name] = sha256_file(selection_protocol_path)
    output_hashes = {path.name: sha256_file(path) for path in paths.values()}
    duration = time.perf_counter() - started
    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "generated_at_utc": generated_at.isoformat(),
        "duration_seconds": duration,
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output_dir),
        "counts": {
            "eligible_origin_pairs": len(data.base.records),
            "unique_oof_validation_pairs": predictions["pair_id"].nunique(),
            "prediction_rows_long": len(predictions),
            "full_feature_count_for_mode": len(data.base.feature_columns),
            "reduced_ridge_feature_count": len(data.reduced_feature_columns),
            "base_signal_count": len(data.base_signal_names),
            "observed_mode_count": data.mode_codes.nunique(),
            "fold_count": len(fold_ids),
        },
        "models_reported": list(MODEL_NAMES),
        "targets_reported": list(TARGET_COLUMNS),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "safety": {
            "external_2026_read": False,
            "llm_calls_made": 0,
            "random_split_used": False,
            "preprocessing_fit_scope": "TRAIN_FOLD_ONLY",
            "outer_validation_used_for_ridge_alpha": False,
            "label_columns_in_x": [],
            "p2_v1_artifacts_written": False,
            "model_selection_performed": False,
            "eligibility_assessment_only": True,
            "selected_model_written": False,
        },
    }
    _write_json(output_dir / "run_manifest_v1.json", manifest)
    return ResidualRunResult(
        output_dir=output_dir,
        predictions=predictions,
        fold_metrics=fold_metrics,
        overall_metrics=overall,
        mode_coverage=mode_coverage,
        manifest=manifest,
    )
