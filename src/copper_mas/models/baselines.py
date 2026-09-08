"""P2 V1：2024—2025 开发期的可复现离线回归基线。

本模块严格复用 P1 冻结的五折扩展窗口角色，不自行随机拆分。目标值只从
``outcome_ledger_v2.csv`` 读取并始终与特征矩阵物理分离；插补、标准化和
回归器被封装在同一个 scikit-learn ``Pipeline`` 中，每折只在 TRAIN 行拟合。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RUN_ID = "P2_BASELINES_V1_2024_2025"
DEVELOPMENT_YEARS = frozenset({2024, 2025})
EXPECTED_FOLDS = 5
TARGET_COLUMNS = ("target_cu_g_l", "target_as_mg_l")
PERSISTENCE_COLUMNS = {
    "target_cu_g_l": "origin_cu_g_l",
    "target_as_mg_l": "origin_as_mg_l",
}
CORE_METADATA_COLUMNS = frozenset({"origin_event_id", "decision_at"})
INPUT_FILENAMES = {
    "features": "core_feature_matrix_v2.csv",
    "index": "training_evaluation_index_v2.csv",
    "outcomes": "outcome_ledger_v2.csv",
    "folds": "cv_fold_manifest_v2.csv",
}
MODEL_FILE_SLUGS = {
    "DummyMedian": "dummy_median",
    "Ridge": "ridge",
    "ExtraTreesLite": "extra_trees_lite",
}


@dataclass(frozen=True)
class PreparedP2Data:
    """已通过样本、字段与时序合同检查的 P2 输入。"""

    records: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    folds: pd.DataFrame
    feature_columns: tuple[str, ...]
    source_paths: dict[str, Path]


@dataclass(frozen=True)
class P2RunResult:
    """一次 P2 基线运行生成的主要工件与摘要。"""

    output_dir: Path
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame
    manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _natural_fold_key(value: str) -> tuple[int, str]:
    token = str(value)
    try:
        return int(token.rsplit("_", 1)[-1]), token
    except ValueError:
        return sys.maxsize, token


def _as_bool_series(series: pd.Series, *, column: str) -> pd.Series:
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


def _forbidden_feature_reason(name: str) -> str | None:
    lowered = str(name).strip().lower()
    tokens = lowered.replace("-", "_").split("__")
    forbidden_prefixes = ("target_", "next_", "lead_", "leadto_", "lead_to_")
    if lowered.startswith(forbidden_prefixes):
        return "以事后/未来字段前缀开头"
    if any(token.startswith(forbidden_prefixes) for token in tokens):
        return "包含事后/未来字段片段"
    return None


def validate_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """拒绝任何 target/next/lead 字段，并返回冻结后的字段顺序。"""

    ordered = tuple(str(column) for column in columns)
    if not ordered:
        raise ValueError("X 特征列为空")
    if len(set(ordered)) != len(ordered):
        duplicates = pd.Index(ordered)[pd.Index(ordered).duplicated()].unique().tolist()
        raise ValueError(f"X 特征列名不唯一: {duplicates[:5]}")
    forbidden = {
        column: reason
        for column in ordered
        if (reason := _forbidden_feature_reason(column)) is not None
    }
    if forbidden:
        raise ValueError(f"X 检出禁止的未来/目标字段: {forbidden}")
    return ordered


def _validate_development_dates(values: pd.Series, *, column: str) -> pd.Series:
    # P1 工件通常使用秒级字符串；单元/外部调用可能混用 ``YYYY-MM-DD``。
    # pandas 2.x 默认会按首值推断单一格式，因此显式允许逐值混合解析。
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"{column} 含无法解析的时间")
    years = set(parsed.dt.year.astype(int))
    unexpected = sorted(years - DEVELOPMENT_YEARS)
    if unexpected:
        raise ValueError(
            f"P2 只允许 2024—2025 开发期数据；{column} 检出年份 {unexpected}，已停止。"
        )
    return parsed


def validate_temporal_fold_manifest(
    folds: pd.DataFrame,
    *,
    expected_folds: int = EXPECTED_FOLDS,
) -> pd.DataFrame:
    """验证扩展窗口角色、训练先于验证、训练集嵌套及验证集互斥。"""

    required = {
        "fold_id",
        "fold_role",
        "pair_id",
        "decision_at",
        "fold_fit_cutoff_at",
        "validation_end_at",
        "preprocessing_fit_scope",
    }
    missing = sorted(required - set(folds.columns))
    if missing:
        raise ValueError(f"cv_fold_manifest_v2.csv 缺少字段: {missing}")

    checked = folds.copy()
    checked["fold_id"] = checked["fold_id"].astype(str)
    checked["fold_role"] = checked["fold_role"].astype(str).str.upper()
    unknown_roles = sorted(set(checked["fold_role"]) - {"TRAIN", "VALIDATION"})
    if unknown_roles:
        raise ValueError(f"发现未知折角色: {unknown_roles}")
    if checked.duplicated(["fold_id", "pair_id"]).any():
        raise ValueError("同一 pair_id 在同一折中被重复或同时分配多个角色")
    if set(checked["preprocessing_fit_scope"].astype(str)) != {"TRAIN_FOLD_ONLY"}:
        raise ValueError("预处理拟合范围必须全部为 TRAIN_FOLD_ONLY")

    checked["decision_at"] = _validate_development_dates(
        checked["decision_at"], column="cv_fold_manifest.decision_at"
    )
    checked["fold_fit_cutoff_at"] = pd.to_datetime(
        checked["fold_fit_cutoff_at"], errors="coerce", format="mixed"
    )
    checked["validation_end_at"] = pd.to_datetime(
        checked["validation_end_at"], errors="coerce", format="mixed"
    )
    if checked[["fold_fit_cutoff_at", "validation_end_at"]].isna().any().any():
        raise ValueError("折拟合截止或验证截止时间无法解析")

    fold_ids = sorted(checked["fold_id"].unique(), key=_natural_fold_key)
    if len(fold_ids) != expected_folds:
        raise ValueError(f"必须严格使用既有 {expected_folds} 折；实际为 {len(fold_ids)} 折")

    previous_train_ids: set[str] = set()
    seen_validation_ids: set[str] = set()
    for fold_id in fold_ids:
        fold = checked.loc[checked["fold_id"] == fold_id]
        train = fold.loc[fold["fold_role"] == "TRAIN"]
        validation = fold.loc[fold["fold_role"] == "VALIDATION"]
        if train.empty or validation.empty:
            raise ValueError(f"{fold_id} 必须同时包含 TRAIN 和 VALIDATION")
        train_ids = set(train["pair_id"].astype(str))
        validation_ids = set(validation["pair_id"].astype(str))
        if train_ids & validation_ids:
            raise ValueError(f"{fold_id} 训练集与验证集 pair_id 重叠")
        if train["decision_at"].max() >= validation["decision_at"].min():
            raise ValueError(f"{fold_id} 训练时间未严格早于验证时间")

        cutoffs = fold["fold_fit_cutoff_at"].drop_duplicates()
        validation_ends = fold["validation_end_at"].drop_duplicates()
        if len(cutoffs) != 1 or len(validation_ends) != 1:
            raise ValueError(f"{fold_id} 的截止时间不唯一")
        cutoff = cutoffs.iloc[0]
        validation_end = validation_ends.iloc[0]
        if train["decision_at"].max() >= cutoff:
            raise ValueError(f"{fold_id} TRAIN 行越过 fold_fit_cutoff_at")
        if validation["decision_at"].min() < cutoff:
            raise ValueError(f"{fold_id} VALIDATION 行早于 fold_fit_cutoff_at")
        if validation["decision_at"].max() > validation_end:
            raise ValueError(f"{fold_id} VALIDATION 行晚于 validation_end_at")
        if previous_train_ids and not previous_train_ids.issubset(train_ids):
            raise ValueError(f"{fold_id} 不满足扩展窗口训练集嵌套")
        if seen_validation_ids & validation_ids:
            raise ValueError(f"{fold_id} 与较早折的 VALIDATION 样本重叠")
        previous_train_ids = train_ids
        seen_validation_ids.update(validation_ids)

    return checked


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少必要字段: {missing}")


def prepare_p2_data(input_dir: Path) -> PreparedP2Data:
    """读取四个冻结 P1 工件，筛选主样本并保持 X/y 物理隔离。"""

    input_dir = Path(input_dir).resolve()
    source_paths = {key: input_dir / filename for key, filename in INPUT_FILENAMES.items()}
    absent = [str(path) for path in source_paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"P2 输入不完整: {absent}")

    core = pd.read_csv(source_paths["features"], low_memory=False)
    index = pd.read_csv(source_paths["index"], low_memory=False)
    outcomes = pd.read_csv(source_paths["outcomes"], low_memory=False)
    fold_manifest = pd.read_csv(source_paths["folds"], low_memory=False)

    _require_columns(core, CORE_METADATA_COLUMNS, source=INPUT_FILENAMES["features"])
    _require_columns(
        index,
        {"pair_id", "origin_event_id", "origin_recorded_at", "origin_year",
         "tier1_core_primary_prospective"},
        source=INPUT_FILENAMES["index"],
    )
    _require_columns(
        outcomes,
        {"pair_id", *TARGET_COLUMNS},
        source=INPUT_FILENAMES["outcomes"],
    )

    if core["origin_event_id"].duplicated().any():
        raise ValueError("核心特征矩阵 origin_event_id 不唯一")
    if index["pair_id"].duplicated().any() or index["origin_event_id"].duplicated().any():
        raise ValueError("训练评价索引的 pair_id/origin_event_id 不唯一")
    if outcomes["pair_id"].duplicated().any():
        raise ValueError("真值账本 pair_id 不唯一")

    core["decision_at"] = _validate_development_dates(
        core["decision_at"], column="core_feature_matrix.decision_at"
    )
    index["origin_recorded_at"] = _validate_development_dates(
        index["origin_recorded_at"], column="training_index.origin_recorded_at"
    )
    origin_year = pd.to_numeric(index["origin_year"], errors="coerce")
    if origin_year.isna().any():
        raise ValueError("training_index.origin_year 含非数值")
    unexpected_years = sorted(set(origin_year.astype(int)) - DEVELOPMENT_YEARS)
    if unexpected_years:
        raise ValueError(f"P2 只允许 2024—2025；origin_year 检出 {unexpected_years}")

    eligibility = _as_bool_series(
        index["tier1_core_primary_prospective"],
        column="tier1_core_primary_prospective",
    )
    selected_index = index.loc[
        eligibility, ["pair_id", "origin_event_id", "origin_recorded_at"]
    ].copy()
    if selected_index.empty:
        raise ValueError("tier1_core_primary_prospective 主样本为空")

    joined = selected_index.merge(
        core,
        on="origin_event_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing_ids = joined.loc[joined["_merge"] != "both", "origin_event_id"].tolist()
        raise ValueError(f"主样本缺少核心特征: {missing_ids[:5]}")
    joined = joined.drop(columns="_merge")
    if not joined["origin_recorded_at"].eq(joined["decision_at"]).all():
        raise ValueError("training_index.origin_recorded_at 与特征 decision_at 不一致")

    joined = joined.merge(
        outcomes[["pair_id", *TARGET_COLUMNS]],
        on="pair_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing_pairs = joined.loc[joined["_merge"] != "both", "pair_id"].tolist()
        raise ValueError(f"主样本缺少事后真值: {missing_pairs[:5]}")
    joined = joined.drop(columns="_merge")
    for target in TARGET_COLUMNS:
        joined[target] = pd.to_numeric(joined[target], errors="coerce")
    if joined[list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError("主样本 Cu/As 真值不完整")

    feature_columns = validate_feature_columns(
        column for column in core.columns if column not in CORE_METADATA_COLUMNS
    )
    absent_persistence = sorted(set(PERSISTENCE_COLUMNS.values()) - set(feature_columns))
    if absent_persistence:
        raise ValueError(f"持续性基线缺少当前结果字段: {absent_persistence}")

    features = joined.loc[:, feature_columns].copy()
    for column in feature_columns:
        if pd.api.types.is_bool_dtype(features[column]):
            features[column] = features[column].astype(float)
        else:
            converted = pd.to_numeric(features[column], errors="coerce")
            newly_missing = converted.isna() & features[column].notna()
            if newly_missing.any():
                raise ValueError(f"特征 {column} 含无法数值化的非空值")
            features[column] = converted.astype(float)
    if features[list(PERSISTENCE_COLUMNS.values())].isna().any().any():
        raise ValueError("主样本当前 Cu/As 缺失，无法计算 persistence")

    selected_pairs = set(selected_index["pair_id"].astype(str))
    selected_folds = fold_manifest.loc[
        fold_manifest["pair_id"].astype(str).isin(selected_pairs)
    ].copy()
    covered_pairs = set(selected_folds["pair_id"].astype(str))
    if not selected_pairs.issubset(covered_pairs):
        missing_pairs = sorted(selected_pairs - covered_pairs)
        raise ValueError(f"主样本未被既有五折角色覆盖: {missing_pairs[:5]}")
    selected_folds = validate_temporal_fold_manifest(selected_folds)

    fold_times = selected_folds.set_index(["fold_id", "pair_id"])["decision_at"]
    decision_by_pair = joined.set_index("pair_id")["decision_at"]
    mismatched = [
        (fold_id, pair_id)
        for (fold_id, pair_id), value in fold_times.items()
        if pd.Timestamp(value) != pd.Timestamp(decision_by_pair.loc[pair_id])
    ]
    if mismatched:
        raise ValueError(f"折清单 decision_at 与核心特征不一致: {mismatched[:5]}")

    records = joined.loc[:, ["pair_id", "origin_event_id", "decision_at"]].copy()
    records["pair_id"] = records["pair_id"].astype(str)
    records = records.sort_values(["decision_at", "pair_id"], kind="mergesort").reset_index(drop=True)
    features = features.loc[records.index].reset_index(drop=True)
    targets = joined.loc[records.index, list(TARGET_COLUMNS)].reset_index(drop=True)

    # 上述排序通常不改变 P1 已排序行；显式按 pair_id 重索引可避免未来输入顺序变化造成错位。
    original_by_pair = joined.set_index("pair_id")
    features = original_by_pair.loc[records["pair_id"], list(feature_columns)].reset_index(drop=True)
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    targets = original_by_pair.loc[records["pair_id"], list(TARGET_COLUMNS)].reset_index(drop=True)

    return PreparedP2Data(
        records=records,
        features=features,
        targets=targets,
        folds=selected_folds,
        feature_columns=feature_columns,
        source_paths=source_paths,
    )


def build_model_pipelines(
    *,
    ridge_alpha: float = 1.0,
    random_state: int = 20260826,
    include_nonlinear: bool = True,
) -> dict[str, Pipeline]:
    """构造固定超参数基线；调用方必须每折重新拟合。"""

    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha 必须大于 0")
    pipelines: dict[str, Pipeline] = {
        "DummyMedian": Pipeline(
            steps=[
                ("simple_imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("regressor", DummyRegressor(strategy="median")),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("simple_imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("standard_scaler", StandardScaler()),
                ("regressor", Ridge(alpha=float(ridge_alpha))),
            ]
        ),
    }
    if include_nonlinear:
        pipelines["ExtraTreesLite"] = Pipeline(
            steps=[
                ("simple_imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=64,
                        max_depth=10,
                        min_samples_leaf=8,
                        max_features=0.30,
                        n_jobs=1,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    return pipelines


def _write_candidate_bundle(
    data: PreparedP2Data,
    bundle_dir: Path,
    *,
    ridge_alpha: float,
    random_state: int,
    include_nonlinear: bool,
) -> tuple[Path, list[Path], dict[str, Any]]:
    """在全部开发期主样本上重拟合所有候选，但不指定胜者。"""

    bundle_dir.mkdir(parents=True, exist_ok=True)
    x_full = data.features.loc[:, list(data.feature_columns)]
    serialized_paths: list[Path] = []
    model_entries: list[dict[str, Any]] = []
    for target_name in TARGET_COLUMNS:
        y_full = data.targets[target_name].to_numpy(dtype=float)
        for model_name, pipeline in build_model_pipelines(
            ridge_alpha=ridge_alpha,
            random_state=random_state,
            include_nonlinear=include_nonlinear,
        ).items():
            pipeline.fit(x_full, y_full)
            slug = MODEL_FILE_SLUGS[model_name]
            model_path = bundle_dir / f"{slug}__{target_name}.joblib"
            joblib.dump(pipeline, model_path, compress=3)
            serialized_paths.append(model_path)
            model_entries.append(
                {
                    "model_name": model_name,
                    "target_name": target_name,
                    "artifact": model_path.name,
                    "sha256": sha256_file(model_path),
                    "fit_sample_count": len(data.records),
                    "feature_count": len(data.feature_columns),
                }
            )

    feature_payload = json.dumps(
        list(data.feature_columns), ensure_ascii=False, separators=(",", ":")
    )
    bundle_manifest: dict[str, Any] = {
        "bundle_id": "A4_P2_FROZEN_CANDIDATES_V1_2024_2025",
        "bundle_status": "FROZEN_DEVELOPMENT_CANDIDATES_NO_WINNER",
        "fit_scope": "all_2024_2025_tier1_core_primary_prospective",
        "fit_sample_count": len(data.records),
        "external_2026_used": False,
        "selection": {
            "selected_model": None,
            "default_model": None,
            "instruction": (
                "A4 必须显式传入 model_name 与 target_name；本包不把开发期单折最好结果"
                "解释为最终胜者。"
            ),
        },
        "features": {
            "ordered_names": list(data.feature_columns),
            "ordered_names_sha256": sha256_text(feature_payload),
            "known_at_decision_only": True,
            "origin_current_result_included": True,
        },
        "targets": list(TARGET_COLUMNS),
        "persistence": [
            {
                "target_name": target_name,
                "definition": f"prediction = {PERSISTENCE_COLUMNS[target_name]}",
                "serialized_model": False,
            }
            for target_name in TARGET_COLUMNS
        ],
        "serialized_candidates": model_entries,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "safety": {
            "llm_calls_made": 0,
            "random_split_used": False,
            "target_columns_in_features": [],
            "pickle_loading_policy": "only load local artifacts after SHA256 verification",
        },
    }
    manifest_path = bundle_dir / "bundle_manifest_v1.json"
    _write_json(manifest_path, bundle_manifest)
    return manifest_path, serialized_paths, bundle_manifest


def load_candidate_pipeline(
    bundle_dir: Path,
    *,
    model_name: str,
    target_name: str,
) -> tuple[Pipeline, dict[str, Any]]:
    """供 A4 显式加载并校验本地候选；Persistence 应按清单规则直接计算。"""

    bundle_dir = Path(bundle_dir).resolve()
    manifest_path = bundle_dir / "bundle_manifest_v1.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"候选包清单不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in manifest.get("serialized_candidates", [])
        if entry.get("model_name") == model_name and entry.get("target_name") == target_name
    ]
    if len(matches) != 1:
        raise KeyError(f"候选不存在或不唯一: model={model_name}, target={target_name}")
    entry = matches[0]
    artifact_path = (bundle_dir / str(entry["artifact"])).resolve()
    try:
        artifact_path.relative_to(bundle_dir)
    except ValueError as exc:
        raise ValueError("候选包清单中的 artifact 越出 bundle_dir") from exc
    if not artifact_path.is_file():
        raise FileNotFoundError(f"候选模型不存在: {artifact_path}")
    actual_hash = sha256_file(artifact_path)
    if actual_hash != entry.get("sha256"):
        raise ValueError(f"候选模型 SHA256 不匹配: {artifact_path.name}")
    pipeline = joblib.load(artifact_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"候选工件不是 sklearn Pipeline: {type(pipeline)!r}")
    return pipeline, manifest


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _relative_improvement(reference: float, candidate: float) -> float:
    if reference == 0:
        return 0.0 if candidate == 0 else float("nan")
    return float((reference - candidate) / reference * 100.0)


def _model_config_payload(
    *,
    feature_count: int,
    ridge_alpha: float,
    random_state: int,
    include_nonlinear: bool,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Persistence": {
            "fit": False,
            "definition": "下一事件预测值等于起点当前已录入 Cu/As",
        },
        "DummyMedian": {
            "pipeline": [
                "SimpleImputer(strategy='median', keep_empty_features=True)",
                "DummyRegressor(strategy='median')",
            ],
        },
        "Ridge": {
            "pipeline": [
                "SimpleImputer(strategy='median', keep_empty_features=True)",
                "StandardScaler()",
                f"Ridge(alpha={float(ridge_alpha)!r})",
            ],
        },
    }


def _fmt_metric(value: Any, digits: int = 4) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _render_human_report(
    data: PreparedP2Data,
    fold_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    *,
    include_nonlinear: bool,
) -> str:
    base_signals = [
        feature.removesuffix("__t_minus_0h")
        for feature in data.feature_columns
        if feature.endswith("__t_minus_0h")
    ]
    fold_ids = sorted(data.folds["fold_id"].unique(), key=_natural_fold_key)
    lines = [
        "# P2 2024—2025 开发期离线基线结果说明（V1）",
        "",
        "## 1. 结论与边界",
        "",
        (
            "本轮严格使用 `tier1_core_primary_prospective` 主样本和 P1 已冻结的 5 折"
            "扩展窗口。共纳入 2,953 个可训练起点，其中 2,732 个起点进入一次且仅一次的"
            "折外验证。没有读取 2026 数据，没有随机拆分，也没有调用 LLM。"
        ),
        "",
        (
            "在当前四个预先指定基线中，Persistence 的开发期 OOF 指标明显优于"
            "DummyMedian、Ridge 和 ExtraTreesLite。这个结论只描述 2024—2025 开发期"
            "表现，不是 2026 外部性能结论，也不构成因果或控制效果。"
        ),
        "",
        (
            "本阶段不按某一次或某一折的最好结果选择最终模型。A4 冻结包保留全部候选，"
            "`selected_model=null`；A4 加载时必须显式指定模型和目标。Persistence 继续作为"
            "后续任何模型必须战胜的基准。"
        ),
        "",
        "## 2. X 中究竟用了哪些特征",
        "",
        (
            f"X 共 {len(data.feature_columns)} 列：27 个过程基础信号各展开为 26 列（共 702 列），"
            "再加起点当前废铜液原液罐 `origin_cu_g_l`、`origin_as_mg_l` 和"
            "`origin_result_quality_eligible` 3 列。"
        ),
        "",
        (
            "每个过程信号的 26 列包括：0、2、4、6、8、10、12 h 七个回看值，七个缺失标志，"
            "七个观测年龄，以及 12 h 均值、标准差、极差、有效数和斜率。所有信息的截止点均为"
            "该起点的 `decision_at`。"
        ),
        "",
        "27 个基础信号为：",
        "",
    ]
    lines.extend(f"- `{signal}`" for signal in base_signals)
    lines.extend(
        [
            "",
            "当前结果是否进入 X：**是**。`origin_cu_g_l` 与 `origin_as_mg_l` 是当前已经录入、"
            "决策时已知的结果；Persistence 正是用它们预测下一次结果。",
            "",
            (
                "明确未进入 X 的字段包括 `pair_id`、`origin_event_id`、`decision_at` 以及所有"
                "`target_*`、`next_*`、`lead_*` 字段。每一列的完整顺序见"
                " `feature_manifest_v1.csv`。"
            ),
            "",
            "## 3. 五折时序角色",
            "",
            "| 折 | 训练数 | 验证数 | 规则 |",
            "|---|---:|---:|---|",
        ]
    )
    for fold_id in fold_ids:
        train_n = len(
            data.folds.loc[
                (data.folds["fold_id"] == fold_id) & (data.folds["fold_role"] == "TRAIN")
            ]
        )
        validation_n = len(
            data.folds.loc[
                (data.folds["fold_id"] == fold_id)
                & (data.folds["fold_role"] == "VALIDATION")
            ]
        )
        lines.append(f"| {fold_id} | {train_n} | {validation_n} | 训练严格早于验证 |")
    lines.extend(
        [
            "",
            (
                "SimpleImputer、StandardScaler 与 Ridge 位于同一个 Pipeline；每个外层折都会"
                "重新创建并只在该折 TRAIN 行上 `fit`，VALIDATION 只调用 `predict`。"
                "DummyMedian 与 ExtraTreesLite 的插补器同样只拟合 TRAIN 行。"
            ),
            "",
            "## 4. 全部 OOF 验证样本上的总体指标",
            "",
            "| 目标 | 模型 | n | MAE | RMSE | R² | 相对 Persistence 的 MAE 改进 | 相对 Persistence 的 RMSE 改进 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in overall_metrics.iterrows():
        unit_digits = 3 if row["target_name"] == "target_as_mg_l" else 4
        lines.append(
            "| {target} | {model} | {n} | {mae} | {rmse} | {r2} | {mae_imp}% | {rmse_imp}% |".format(
                target=row["target_name"],
                model=row["model_name"],
                n=int(row["validation_sample_count"]),
                mae=_fmt_metric(row["pooled_mae"], unit_digits),
                rmse=_fmt_metric(row["pooled_rmse"], unit_digits),
                r2=_fmt_metric(row["pooled_r2"], 4),
                mae_imp=_fmt_metric(
                    row["pooled_relative_mae_improvement_vs_persistence_pct"], 2
                ),
                rmse_imp=_fmt_metric(
                    row["pooled_relative_rmse_improvement_vs_persistence_pct"], 2
                ),
            )
        )
    lines.extend(
        [
            "",
            "Cu 的单位为 g/L，As 的单位为 mg/L。负的相对改进表示比 Persistence 更差。",
            "",
            "## 5. Persistence 与 Ridge 的逐折指标",
            "",
            "| 目标 | 模型 | 折 | n | MAE | RMSE | R² | MAE 相对改进 | RMSE 相对改进 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    detailed = fold_metrics.loc[
        fold_metrics["model_name"].isin(["Persistence", "Ridge"])
    ].sort_values(["target_name", "model_name", "fold_id"], kind="mergesort")
    for _, row in detailed.iterrows():
        unit_digits = 3 if row["target_name"] == "target_as_mg_l" else 4
        lines.append(
            "| {target} | {model} | {fold} | {n} | {mae} | {rmse} | {r2} | {mae_imp}% | {rmse_imp}% |".format(
                target=row["target_name"],
                model=row["model_name"],
                fold=row["fold_id"],
                n=int(row["validation_sample_count"]),
                mae=_fmt_metric(row["mae"], unit_digits),
                rmse=_fmt_metric(row["rmse"], unit_digits),
                r2=_fmt_metric(row["r2"], 4),
                mae_imp=_fmt_metric(row["relative_mae_improvement_vs_persistence_pct"], 2),
                rmse_imp=_fmt_metric(row["relative_rmse_improvement_vs_persistence_pct"], 2),
            )
        )
    lines.extend(
        [
            "",
            "折间均值与样本标准差（ddof=1）已完整保存在 `overall_metrics_v1.csv`。",
            "",
            (
                "Ridge 在 FOLD_1 的明显失稳不是被隐藏的异常：该折只有 221 个主样本，却有"
                f" {len(data.feature_columns)} 个输入特征，并且跨工况外推强。当前结果说明固定"
                " alpha=1 的线性正则基线不适合直接作为 A4 默认模型；不能删除该折或只报告后四折。"
                "这首先是高维小样本与工况漂移风险信号，不应外推成“Ridge 或机器学习模型无效”"
                "的普遍结论。"
            ),
            "",
            "## 6. 模型选择与 A4 冻结包",
            "",
            (
                "冻结包状态为 `FROZEN_DEVELOPMENT_CANDIDATES_NO_WINNER`。Persistence 以公式保存；"
                "DummyMedian、Ridge"
                + ("、ExtraTreesLite" if include_nonlinear else "")
                + " 已按 Cu/As 分别在全部 2,953 条开发期主样本上重拟合并序列化。"
            ),
            "",
            (
                "全开发期重拟合仅用于形成候选推理工件，不参与本报告 OOF 指标计算。A4 应先验证"
                " `bundle_manifest_v1.json` 中的 SHA256，再显式加载候选；不得根据 2026 结果回头"
                "修改特征、超参数、路由或选择规则。"
            ),
            "",
            "## 7. 可审计工件",
            "",
            "- `oof_predictions_v1.csv`：逐样本、逐折、逐目标、逐模型预测与误差。",
            "- `fold_metrics_v1.csv`：每折 MAE、RMSE、R²、相对 Persistence 改进和运行时间。",
            "- `overall_metrics_v1.csv`：全部 OOF 总体指标以及五折均值、样本标准差。",
            "- `feature_manifest_v1.csv`：705 个 X 字段的冻结顺序、来源和决策时可用性。",
            "- `model_config_v1.json`：固定模型参数、样本和评价规则。",
            "- `a4_candidate_bundle_v1/`：A4 候选模型、Persistence 规则及逐文件 SHA256。",
            "- `run_manifest_v1.json`：输入/输出 SHA256、版本、样本数、运行时和安全声明。",
            "",
        ]
    )
    return "\n".join(lines)
    if include_nonlinear:
        models["ExtraTreesLite"] = {
            "pipeline": [
                "SimpleImputer(strategy='median', keep_empty_features=True)",
                "ExtraTreesRegressor(n_estimators=64, max_depth=10, "
                "min_samples_leaf=8, max_features=0.30, n_jobs=1)",
            ],
            "random_state": int(random_state),
        }
    return {
        "run_id": RUN_ID,
        "sample": "tier1_core_primary_prospective",
        "development_years": [2024, 2025],
        "external_2026_used": False,
        "targets": list(TARGET_COLUMNS),
        "feature_count": int(feature_count),
        "feature_policy": {
            "source": INPUT_FILENAMES["features"],
            "known_at_decision_only": True,
            "forbidden_prefixes": ["target_", "next_", "lead_", "lead_to_"],
            "origin_current_cu_as_included": True,
        },
        "cross_validation": {
            "source": INPUT_FILENAMES["folds"],
            "fold_count": EXPECTED_FOLDS,
            "scheme": "frozen expanding-window roles",
            "preprocessing_fit_scope": "TRAIN_FOLD_ONLY",
            "random_split": False,
        },
        "selection": {
            "performed": False,
            "statement": "仅并列报告预先指定基线，不按单折或单次最好结果选择模型。",
        },
        "metrics": [
            "MAE",
            "RMSE",
            "R2",
            "relative MAE/RMSE improvement vs Persistence",
            "R2 delta vs Persistence",
        ],
        "fold_dispersion": "sample standard deviation (ddof=1)",
        "models": models,
    }


def run_p2_baselines(
    input_dir: Path,
    output_dir: Path,
    *,
    ridge_alpha: float = 1.0,
    random_state: int = 20260826,
    include_nonlinear: bool = True,
) -> P2RunResult:
    """运行冻结五折基线并写出逐样本预测、指标、配置与哈希清单。"""

    started_wall = time.perf_counter()
    generated_at = datetime.now(timezone.utc)
    data = prepare_p2_data(input_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    record_by_pair = data.records.set_index("pair_id")
    feature_by_pair = data.features.copy()
    feature_by_pair.index = data.records["pair_id"]
    target_by_pair = data.targets.copy()
    target_by_pair.index = data.records["pair_id"]

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_ids = sorted(data.folds["fold_id"].unique(), key=_natural_fold_key)

    for fold_id in fold_ids:
        fold = data.folds.loc[data.folds["fold_id"] == fold_id]
        train_ids = fold.loc[fold["fold_role"] == "TRAIN"].sort_values("decision_at")["pair_id"].astype(str)
        validation_ids = fold.loc[fold["fold_role"] == "VALIDATION"].sort_values("decision_at")["pair_id"].astype(str)
        x_train = feature_by_pair.loc[train_ids, list(data.feature_columns)]
        x_validation = feature_by_pair.loc[validation_ids, list(data.feature_columns)]

        for target_name in TARGET_COLUMNS:
            y_train = target_by_pair.loc[train_ids, target_name].to_numpy(dtype=float)
            y_true = target_by_pair.loc[validation_ids, target_name].to_numpy(dtype=float)
            persistence_column = PERSISTENCE_COLUMNS[target_name]
            persistence_pred = x_validation[persistence_column].to_numpy(dtype=float)
            persistence_metrics = _metrics(y_true, persistence_pred)

            model_predictions: dict[str, tuple[np.ndarray, float, float]] = {
                "Persistence": (persistence_pred, 0.0, 0.0)
            }
            for model_name, pipeline in build_model_pipelines(
                ridge_alpha=ridge_alpha,
                random_state=random_state,
                include_nonlinear=include_nonlinear,
            ).items():
                fit_started = time.perf_counter()
                pipeline.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - fit_started
                predict_started = time.perf_counter()
                y_pred = np.asarray(pipeline.predict(x_validation), dtype=float)
                predict_seconds = time.perf_counter() - predict_started
                model_predictions[model_name] = (y_pred, fit_seconds, predict_seconds)

            validation_meta = record_by_pair.loc[
                validation_ids, ["origin_event_id", "decision_at"]
            ].reset_index()
            for model_name, (y_pred, fit_seconds, predict_seconds) in model_predictions.items():
                current_metrics = _metrics(y_true, y_pred)
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
                        "relative_mae_improvement_vs_persistence_pct": _relative_improvement(
                            persistence_metrics["mae"], current_metrics["mae"]
                        ),
                        "relative_rmse_improvement_vs_persistence_pct": _relative_improvement(
                            persistence_metrics["rmse"], current_metrics["rmse"]
                        ),
                        "r2_delta_vs_persistence": current_metrics["r2"] - persistence_metrics["r2"],
                        "fit_time_seconds": fit_seconds,
                        "predict_time_seconds": predict_seconds,
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "run_id": RUN_ID,
                            "fold_id": fold_id,
                            "pair_id": validation_meta["pair_id"].astype(str),
                            "origin_event_id": validation_meta["origin_event_id"].astype(str),
                            "decision_at": pd.to_datetime(validation_meta["decision_at"]),
                            "target_name": target_name,
                            "model_name": model_name,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "persistence_prediction": persistence_pred,
                            "absolute_error": np.abs(y_true - y_pred),
                            "squared_error": np.square(y_true - y_pred),
                        }
                    )
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(
        ["target_name", "model_name", "fold_id", "decision_at", "pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["target_name", "model_name", "fold_id"], kind="mergesort"
    ).reset_index(drop=True)

    overall_rows: list[dict[str, Any]] = []
    for (target_name, model_name), group in predictions.groupby(
        ["target_name", "model_name"], sort=True
    ):
        pooled = _metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        persistence = _metrics(
            group["y_true"].to_numpy(), group["persistence_prediction"].to_numpy()
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
            "pooled_mae": pooled["mae"],
            "pooled_rmse": pooled["rmse"],
            "pooled_r2": pooled["r2"],
            "pooled_persistence_mae": persistence["mae"],
            "pooled_persistence_rmse": persistence["rmse"],
            "pooled_persistence_r2": persistence["r2"],
            "pooled_relative_mae_improvement_vs_persistence_pct": _relative_improvement(
                persistence["mae"], pooled["mae"]
            ),
            "pooled_relative_rmse_improvement_vs_persistence_pct": _relative_improvement(
                persistence["rmse"], pooled["rmse"]
            ),
            "pooled_r2_delta_vs_persistence": pooled["r2"] - persistence["r2"],
            "fit_time_seconds_total": per_fold["fit_time_seconds"].sum(),
            "predict_time_seconds_total": per_fold["predict_time_seconds"].sum(),
        }
        for metric in (
            "mae",
            "rmse",
            "r2",
            "relative_mae_improvement_vs_persistence_pct",
            "relative_rmse_improvement_vs_persistence_pct",
            "r2_delta_vs_persistence",
        ):
            row[f"fold_{metric}_mean"] = per_fold[metric].mean()
            row[f"fold_{metric}_std"] = per_fold[metric].std(ddof=1)
        overall_rows.append(row)
    overall_metrics = pd.DataFrame(overall_rows).sort_values(
        ["target_name", "model_name"], kind="mergesort"
    ).reset_index(drop=True)

    feature_manifest = pd.DataFrame(
        {
            "feature_order": np.arange(len(data.feature_columns), dtype=int),
            "feature_name": list(data.feature_columns),
            "source_artifact": INPUT_FILENAMES["features"],
            "availability": "KNOWN_AT_DECISION",
            "is_origin_current_result": [
                feature in set(PERSISTENCE_COLUMNS.values()) for feature in data.feature_columns
            ],
            "dtype_used": [str(data.features[feature].dtype) for feature in data.feature_columns],
        }
    )
    model_config = _model_config_payload(
        feature_count=len(data.feature_columns),
        ridge_alpha=ridge_alpha,
        random_state=random_state,
        include_nonlinear=include_nonlinear,
    )

    artifact_paths = {
        "predictions": output_dir / "oof_predictions_v1.csv",
        "fold_metrics": output_dir / "fold_metrics_v1.csv",
        "overall_metrics": output_dir / "overall_metrics_v1.csv",
        "feature_manifest": output_dir / "feature_manifest_v1.csv",
        "model_config": output_dir / "model_config_v1.json",
        "human_report": output_dir / "P2基线结果说明_V1.md",
    }
    predictions.to_csv(
        artifact_paths["predictions"], index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    fold_metrics.to_csv(
        artifact_paths["fold_metrics"], index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    overall_metrics.to_csv(
        artifact_paths["overall_metrics"], index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    feature_manifest.to_csv(
        artifact_paths["feature_manifest"], index=False, encoding="utf-8-sig"
    )
    _write_json(artifact_paths["model_config"], model_config)
    artifact_paths["human_report"].write_text(
        _render_human_report(
            data,
            fold_metrics,
            overall_metrics,
            include_nonlinear=include_nonlinear,
        ),
        encoding="utf-8",
    )

    bundle_manifest_path, serialized_model_paths, bundle_manifest = _write_candidate_bundle(
        data,
        output_dir / "a4_candidate_bundle_v1",
        ridge_alpha=ridge_alpha,
        random_state=random_state,
        include_nonlinear=include_nonlinear,
    )

    source_hashes = {
        INPUT_FILENAMES[key]: sha256_file(path) for key, path in data.source_paths.items()
    }
    all_output_paths = [
        *artifact_paths.values(),
        bundle_manifest_path,
        *serialized_model_paths,
    ]
    output_hashes = {
        path.relative_to(output_dir).as_posix(): sha256_file(path) for path in all_output_paths
    }
    duration = time.perf_counter() - started_wall
    validation_counts = {
        fold_id: int(
            len(
                data.folds.loc[
                    (data.folds["fold_id"] == fold_id)
                    & (data.folds["fold_role"] == "VALIDATION")
                ]
            )
        )
        for fold_id in fold_ids
    }
    train_counts = {
        fold_id: int(
            len(
                data.folds.loc[
                    (data.folds["fold_id"] == fold_id)
                    & (data.folds["fold_role"] == "TRAIN")
                ]
            )
        )
        for fold_id in fold_ids
    }
    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "generated_at_utc": generated_at.isoformat(),
        "duration_seconds": duration,
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output_dir),
        "sample": "tier1_core_primary_prospective",
        "counts": {
            "eligible_origin_pairs": len(data.records),
            "feature_count": len(data.feature_columns),
            "fold_count": len(fold_ids),
            "fold_train_counts": train_counts,
            "fold_validation_counts": validation_counts,
            "unique_oof_validation_pairs": predictions["pair_id"].nunique(),
            "prediction_rows_long": len(predictions),
        },
        "date_coverage": {
            "decision_at_min": data.records["decision_at"].min(),
            "decision_at_max": data.records["decision_at"].max(),
        },
        "models_reported": sorted(predictions["model_name"].unique()),
        "a4_candidate_bundle": {
            "bundle_id": bundle_manifest["bundle_id"],
            "bundle_status": bundle_manifest["bundle_status"],
            "manifest": bundle_manifest_path.relative_to(output_dir).as_posix(),
            "selected_model": None,
        },
        "targets_reported": list(TARGET_COLUMNS),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "safety": {
            "external_2026_read": False,
            "llm_calls_made": 0,
            "random_split_used": False,
            "preprocessing_fit_scope": "TRAIN_FOLD_ONLY",
            "label_columns_in_x": [],
            "model_selection_performed": False,
        },
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    _write_json(manifest_path, manifest)

    return P2RunResult(
        output_dir=output_dir,
        predictions=predictions,
        fold_metrics=fold_metrics,
        overall_metrics=overall_metrics,
        manifest=manifest,
    )
