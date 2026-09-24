"""P1 V2：构造严格相邻的目标事件索引与物理隔离账本。

本模块只处理 2024—2025 年废铜液原液罐目标事件。原表字段
``sample_datetime`` 在 V2 中正式解释为 ``recorded_at``，且
``available_at = recorded_at``、``sample_at_assumed = recorded_at - 2 h``。

预测时工件（AsOfAdmissionCardV2）只依赖起点当时可见的信息；下一事件
时间、质量和真值仅存在于事后工件中。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from copper_mas.contracts.cards import (
    CONTRACT_ID,
    AsOfAdmissionCardV2,
    EvaluationEligibilityLedgerV2,
    OutcomeLedgerV2,
)
from copper_mas.data.leakage import assert_no_future_information

SCHEMA_VERSION = "2.0"
DEVELOPMENT_YEARS = frozenset({2024, 2025})
ASSUMED_SAMPLE_DELAY = pd.Timedelta(hours=2)

REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        "target_event_id",
        "sample_datetime",
        "target_cu_g_l",
        "target_as_mg_l",
        "target_source_quality_eligible",
        "target_source_conflict_flag",
        "event_time_parse_eligible",
        "source_file",
        "source_sheet",
        "source_row",
    }
)


@dataclass(frozen=True)
class EventDatasetV2:
    """P1 构建结果；各 DataFrame 对应一个独立持久化工件。"""

    event_pair_index: pd.DataFrame
    as_of_admission_cards: pd.DataFrame
    evaluation_ledger: pd.DataFrame
    outcome_ledger: pd.DataFrame
    partition_manifest: pd.DataFrame
    cv_fold_manifest: pd.DataFrame
    cv_fold_summary: pd.DataFrame
    count_difference_summary: pd.DataFrame
    source_event_count: int
    legacy_split_verified: bool | None


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "是"}:
        return True
    if normalized in {"false", "0", "no", "n", "否", ""}:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def _pair_id(origin_event_id: str, target_event_id: str | None) -> str:
    target_token = target_event_id if target_event_id is not None else "FINAL_AUDIT_NO_OUTCOME"
    payload = f"{CONTRACT_ID}|{origin_event_id}|{target_token}".encode()
    return "pair_" + hashlib.sha256(payload).hexdigest()[:20]


def _iso(value: pd.Timestamp | Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ", timespec="seconds")


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prepare_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(source.columns))
    if missing:
        raise ValueError(f"目标事实表缺少必要字段: {missing}")

    frame = source.copy()
    frame["recorded_at"] = pd.to_datetime(frame["sample_datetime"], errors="coerce")
    if frame["recorded_at"].isna().any():
        bad = frame.loc[frame["recorded_at"].isna(), "target_event_id"].astype(str).tolist()
        raise ValueError(f"存在无法解析的录入时间，不能可靠构造相邻序列: {bad[:5]}")

    frame["event_id"] = frame["target_event_id"].astype("string")
    if frame["event_id"].isna().any() or frame["event_id"].str.strip().eq("").any():
        raise ValueError("target_event_id 不能为空")
    if frame["event_id"].duplicated().any():
        duplicates = frame.loc[frame["event_id"].duplicated(False), "event_id"].tolist()
        raise ValueError(f"稳定事件 ID 不唯一: {duplicates[:5]}")

    frame["event_year"] = frame["recorded_at"].dt.year.astype(int)
    unexpected_years = sorted(set(frame["event_year"]) - DEVELOPMENT_YEARS)
    if unexpected_years:
        raise ValueError(
            f"P1 开发集构建器只允许读取 2024—2025；发现年份 {unexpected_years}。为防止提前接触 2026，已停止。"
        )
    if "year" in frame.columns:
        source_year = pd.to_numeric(frame["year"], errors="coerce")
        mismatch = source_year.notna() & source_year.ne(frame["event_year"])
        if mismatch.any():
            ids = frame.loc[mismatch, "event_id"].tolist()
            raise ValueError(f"源 year 与录入时间年份不一致: {ids[:5]}")

    frame["current_cu"] = pd.to_numeric(frame["target_cu_g_l"], errors="coerce")
    frame["current_as"] = pd.to_numeric(frame["target_as_mg_l"], errors="coerce")
    frame["source_quality_eligible"] = frame["target_source_quality_eligible"].map(_as_bool)
    frame["source_conflict"] = frame["target_source_conflict_flag"].map(_as_bool)
    frame["time_parse_eligible"] = frame["event_time_parse_eligible"].map(_as_bool)

    # mergesort 保持稳定；event_id 是冻结的第二排序键。
    frame = frame.sort_values(["recorded_at", "event_id"], kind="mergesort").reset_index(drop=True)
    frame["sequence_index"] = frame.index.astype(int)
    frame["available_at_v2"] = frame["recorded_at"]
    frame["sample_at_assumed_v2"] = frame["recorded_at"] - ASSUMED_SAMPLE_DELAY

    # 合同要求 target_recorded_at > decision_at；相同录入时间虽可稳定排序，但不构成
    # 有效的一步预测，因此 fail closed，而不是跳过其中一条。
    duplicated_times = frame["recorded_at"].duplicated(keep=False)
    if duplicated_times.any():
        rows = frame.loc[duplicated_times, ["event_id", "recorded_at"]].head(5)
        raise ValueError(
            "发现相同 recorded_at 的事件；稳定 ID 已能确定顺序，但合同要求严格正时间跨度，"
            f"不能静默配对或跳过。示例: {rows.to_dict(orient='records')}"
        )
    return frame


def _build_pair_index(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count = len(events)
    for index, origin in events.iterrows():
        has_target = index + 1 < count
        target = events.iloc[index + 1] if has_target else None
        target_id = str(target["event_id"]) if target is not None else None
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_id": CONTRACT_ID,
                "sequence_index": int(origin["sequence_index"]),
                "pair_id": _pair_id(str(origin["event_id"]), target_id),
                "origin_event_id": str(origin["event_id"]),
                "origin_recorded_at": _iso(origin["recorded_at"]),
                "origin_available_at": _iso(origin["available_at_v2"]),
                "origin_sample_at_assumed": _iso(origin["sample_at_assumed_v2"]),
                "target_event_id": target_id,
                "target_recorded_at": _iso(target["recorded_at"]) if target is not None else None,
                "target_available_at": _iso(target["available_at_v2"]) if target is not None else None,
                "target_sample_at_assumed": (_iso(target["sample_at_assumed_v2"]) if target is not None else None),
                "origin_year": int(origin["event_year"]),
                "target_year": int(target["event_year"]) if target is not None else pd.NA,
                "cross_year_pair": (
                    bool(origin["event_year"] != target["event_year"]) if target is not None else False
                ),
                "pair_status": "LABELED_STRICT_ADJACENT" if has_target else "FINAL_EVENT_AUDIT_NO_LABEL",
            }
        )
    return pd.DataFrame(rows)


def _build_as_of_admission_cards(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        result_complete = bool(pd.notna(event["current_cu"]) and pd.notna(event["current_as"]))
        parse_ok = bool(event["time_parse_eligible"])
        warnings: list[str] = []
        reasons: list[str] = []

        if not result_complete:
            reasons.append("CURRENT_RESULT_INCOMPLETE")
        if not parse_ok:
            reasons.append("CURRENT_SOURCE_TIME_FLAG_INELIGIBLE")
        if not bool(event["source_quality_eligible"]):
            warnings.append("CURRENT_SOURCE_QUALITY_WARNING")
        if bool(event["source_conflict"]):
            warnings.append("CURRENT_SOURCE_CONFLICT_WARNING")

        if reasons:
            status = "REJECTED"
        elif warnings:
            status = "WARNING"
        else:
            status = "ADMITTED"

        card = AsOfAdmissionCardV2(
            origin_event_id=str(event["event_id"]),
            decision_at=event["recorded_at"].to_pydatetime(),
            feature_cutoff_at=event["recorded_at"].to_pydatetime(),
            admission_status=status,
            reason_codes=tuple(reasons),
            feature_group_counts={"current_waste_tank_result": int(result_complete)},
            missing_feature_groups=(),
            source_quality_warnings=tuple(warnings),
        )
        assert_no_future_information(card, decision_at=card.decision_at)
        payload = card.model_dump(mode="json")
        payload["reason_codes"] = _json_cell(payload["reason_codes"])
        payload["feature_group_counts"] = _json_cell(payload["feature_group_counts"])
        payload["missing_feature_groups"] = _json_cell(payload["missing_feature_groups"])
        payload["source_quality_warnings"] = _json_cell(payload["source_quality_warnings"])
        rows.append(payload)
    return pd.DataFrame(rows)


def _build_post_outcome_ledgers(events: pd.DataFrame, pair_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []

    for index in range(len(events) - 1):
        origin = events.iloc[index]
        target = events.iloc[index + 1]
        pair_id = str(pair_index.iloc[index]["pair_id"])
        lead_hours = (target["recorded_at"] - origin["recorded_at"]).total_seconds() / 3600
        if lead_hours <= 0:
            raise ValueError(f"相邻事件时间跨度非正: {origin['event_id']} -> {target['event_id']}")

        prospective = bool(target["sample_at_assumed_v2"] > origin["recorded_at"])
        quality_eligible = bool(target["source_quality_eligible"])
        conflict = bool(target["source_conflict"])
        outcome_complete = bool(pd.notna(target["current_cu"]) and pd.notna(target["current_as"]))
        reasons: list[str] = []
        if not prospective:
            reasons.append("NON_PROSPECTIVE_UNDER_2H_ASSUMPTION")
        if not quality_eligible:
            reasons.append("TARGET_SOURCE_QUALITY_INELIGIBLE")
        if conflict:
            reasons.append("TARGET_SOURCE_CONFLICT")
        if not outcome_complete:
            reasons.append("TARGET_OUTCOME_INCOMPLETE")
        evaluation_eligible = prospective and quality_eligible and not conflict and outcome_complete

        ledger = EvaluationEligibilityLedgerV2(
            pair_id=pair_id,
            origin_event_id=str(origin["event_id"]),
            target_event_id=str(target["event_id"]),
            decision_at=origin["recorded_at"].to_pydatetime(),
            target_recorded_at=target["recorded_at"].to_pydatetime(),
            target_sample_at_assumed=target["sample_at_assumed_v2"].to_pydatetime(),
            lead_to_recorded_hours=float(lead_hours),
            prospectively_sampled=prospective,
            target_source_quality_eligible=quality_eligible,
            evaluation_eligible=evaluation_eligible,
            evaluation_reason_codes=tuple(reasons),
        )
        payload = ledger.model_dump(mode="json")
        payload["evaluation_reason_codes"] = _json_cell(payload["evaluation_reason_codes"])
        evaluation_rows.append(payload)

        if outcome_complete:
            outcome = OutcomeLedgerV2(
                pair_id=pair_id,
                target_cu_g_l=float(target["current_cu"]),
                target_as_mg_l=float(target["current_as"]),
            )
            outcome_rows.append(outcome.model_dump(mode="json"))

    return pd.DataFrame(evaluation_rows), pd.DataFrame(outcome_rows)


def _verify_legacy_split(events: pd.DataFrame, legacy_split_path: Path | None) -> bool | None:
    if legacy_split_path is None:
        return None
    legacy = pd.read_csv(legacy_split_path, low_memory=False)
    required = {"target_event_id", "year", "split_role"}
    missing = required.difference(legacy.columns)
    if missing:
        raise ValueError(f"V1 split_manifest 缺少字段: {sorted(missing)}")
    current_ids = set(events["event_id"].astype(str))
    legacy_ids = set(legacy["target_event_id"].astype(str))
    if current_ids != legacy_ids:
        raise ValueError(
            "V1 split_manifest 与当前规范目标事件集合不同，不能生成可靠差异摘要: "
            f"仅当前={len(current_ids - legacy_ids)}, 仅V1={len(legacy_ids - current_ids)}"
        )
    observed_roles = {
        int(year): set(group["split_role"].astype(str))
        for year, group in legacy.groupby(pd.to_numeric(legacy["year"], errors="raise"))
    }
    expected_roles = {
        2024: {"DEVELOPMENT_TRAIN_AND_INNER_CV"},
        2025: {"LOCKED_TEMPORAL_TEST"},
    }
    if observed_roles != expected_roles:
        raise ValueError(
            f"V1 split_manifest 年度角色与待迁移合同不一致: observed={observed_roles}, expected={expected_roles}"
        )
    return True


def _build_partition_manifest(
    pair_index: pd.DataFrame,
    as_of_cards: pd.DataFrame,
    evaluation_ledger: pd.DataFrame,
    outcome_ledger: pd.DataFrame,
) -> pd.DataFrame:
    admission_by_origin = as_of_cards.set_index("origin_event_id")["admission_status"].to_dict()
    eligible_by_pair = evaluation_ledger.set_index("pair_id")["evaluation_eligible"].to_dict()
    outcome_pairs = set(outcome_ledger["pair_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for row in pair_index.itertuples(index=False):
        has_label = row.pair_id in outcome_pairs
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_id": CONTRACT_ID,
                "partition_id": f"DEV_{row.origin_year}",
                "partition_role": "DEVELOPMENT",
                "origin_event_id": row.origin_event_id,
                "pair_id": row.pair_id,
                "decision_at": row.origin_recorded_at,
                "origin_year": row.origin_year,
                "admission_status": admission_by_origin[row.origin_event_id],
                "has_outcome": has_label,
                "primary_evaluation_eligible": bool(eligible_by_pair.get(row.pair_id, False)),
                "row_usage": (
                    "PRIMARY_PROSPECTIVE_MODELING_POOL"
                    if bool(eligible_by_pair.get(row.pair_id, False))
                    else "AUDIT_OR_SENSITIVITY_ONLY"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_expanding_window_folds(
    pair_index: pd.DataFrame,
    evaluation_ledger: pd.DataFrame,
    *,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_splits != 5:
        raise ValueError("活动 V2 合同固定生成 5 折扩展窗口；n_splits 必须为 5")

    eligible = evaluation_ledger.loc[evaluation_ledger["evaluation_eligible"].map(_as_bool)].copy()
    lookup = pair_index[["pair_id", "sequence_index", "origin_recorded_at", "target_recorded_at"]].copy()
    eligible = eligible.merge(lookup, on="pair_id", how="left", validate="one_to_one")
    eligible = eligible.sort_values("sequence_index", kind="mergesort").reset_index(drop=True)
    if len(eligible) <= n_splits:
        raise ValueError(f"主前瞻评价样本 {len(eligible)} 条，不足以生成 {n_splits} 折")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for fold_number, (train_positions, validation_positions) in enumerate(splitter.split(eligible), start=1):
        train = eligible.iloc[train_positions]
        validation = eligible.iloc[validation_positions]
        validation_start = pd.Timestamp(validation["decision_at"].min())
        validation_end = pd.Timestamp(validation["decision_at"].max())
        latest_train_outcome = pd.Timestamp(train["target_recorded_at_y"].max())
        if latest_train_outcome > validation_start:
            raise ValueError(
                f"FOLD_{fold_number} 训练真值在验证开始后才可用: {latest_train_outcome} > {validation_start}"
            )

        fold_id = f"FOLD_{fold_number}"
        for role, subset in (("TRAIN", train), ("VALIDATION", validation)):
            for row in subset.itertuples(index=False):
                manifest_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "contract_id": CONTRACT_ID,
                        "fold_id": fold_id,
                        "fold_role": role,
                        "pair_id": row.pair_id,
                        "origin_event_id": row.origin_event_id,
                        "sequence_index": int(row.sequence_index),
                        "decision_at": row.decision_at,
                        "fold_fit_cutoff_at": _iso(validation_start),
                        "validation_end_at": _iso(validation_end),
                        "preprocessing_fit_scope": "TRAIN_FOLD_ONLY",
                    }
                )
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_id": CONTRACT_ID,
                "fold_id": fold_id,
                "train_count": len(train),
                "validation_count": len(validation),
                "train_start_at": train["decision_at"].min(),
                "train_end_at": train["decision_at"].max(),
                "latest_train_outcome_available_at": _iso(latest_train_outcome),
                "validation_start_at": _iso(validation_start),
                "validation_end_at": _iso(validation_end),
                "temporal_order_check": "PASS",
            }
        )
    return pd.DataFrame(manifest_rows), pd.DataFrame(summary_rows)


def _build_count_difference_summary(
    events: pd.DataFrame,
    pair_index: pd.DataFrame,
    as_of_cards: pd.DataFrame,
    evaluation_ledger: pd.DataFrame,
) -> pd.DataFrame:
    n_events = len(events)
    n_pairs = int(pair_index["target_event_id"].notna().sum())
    n_prospective = int(evaluation_ledger["prospectively_sampled"].map(_as_bool).sum())
    n_primary = int(evaluation_ledger["evaluation_eligible"].map(_as_bool).sum())
    n_asof_accepted = int(as_of_cards["admission_status"].ne("REJECTED").sum())
    n_2024 = int(events["event_year"].eq(2024).sum())
    n_2025 = int(events["event_year"].eq(2025).sum())

    # V1 的 TIER0 规则可由同一事件序列直接复算：前瞻 + 下一真值质量合格。
    # 它不是合法的预测时准入规则，因此与 V2 的 origin-only 准入只做迁移审计，
    # 不宣称两者是同一统计量。
    records = [
        ("目标事件总数", n_events, n_events, "事件真源未改变"),
        ("严格相邻候选事件对", n_pairs, n_pairs, "均不因下一条质量差而跳行"),
        ("主前瞻事件对", n_prospective, n_prospective, "下一假设取样晚于决策时刻"),
        ("主评价合格事件对", n_primary, n_primary, "主前瞻且下一真值完整、来源质量合格"),
        (
            "预测时可接受准入行",
            n_primary,
            n_asof_accepted,
            "V1数值对应含未来条件的TIER0；V2只按起点当时可见状态，二者仅用于迁移审计",
        ),
        ("2024开发事件", n_2024, n_2024, "年度角色不变"),
        ("2025开发事件", 0, n_2025, "V2将2025并入共同开发集"),
        ("2025锁定测试事件", n_2025, 0, "V2不再把2025作为锁定测试"),
        ("最后一条无标签审计行", 1 if n_events else 0, 1 if n_events else 0, "保留但不生成真值账本"),
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "v1_count": v1,
                "v2_count": v2,
                "delta_v2_minus_v1": v2 - v1,
                "interpretation": note,
            }
            for metric, v1, v2, note in records
        ]
    )


def build_event_dataset_v2(
    source: pd.DataFrame,
    *,
    legacy_split_path: Path | None = None,
    n_splits: int = 5,
) -> EventDatasetV2:
    """从内存中的规范目标事实表构建 P1 V2 全部事件层工件。"""

    events = _prepare_source(source)
    if len(events) < 2:
        raise ValueError("至少需要 2 条目标事件才能构造一步预测事件对")
    pair_index = _build_pair_index(events)
    as_of_cards = _build_as_of_admission_cards(events)
    evaluation, outcomes = _build_post_outcome_ledgers(events, pair_index)
    legacy_verified = _verify_legacy_split(events, legacy_split_path)
    partitions = _build_partition_manifest(pair_index, as_of_cards, evaluation, outcomes)
    cv_manifest, cv_summary = _build_expanding_window_folds(pair_index, evaluation, n_splits=n_splits)
    differences = _build_count_difference_summary(events, pair_index, as_of_cards, evaluation)
    return EventDatasetV2(
        event_pair_index=pair_index,
        as_of_admission_cards=as_of_cards,
        evaluation_ledger=evaluation,
        outcome_ledger=outcomes,
        partition_manifest=partitions,
        cv_fold_manifest=cv_manifest,
        cv_fold_summary=cv_summary,
        count_difference_summary=differences,
        source_event_count=len(events),
        legacy_split_verified=legacy_verified,
    )


def load_and_build_event_dataset_v2(
    source_path: Path,
    *,
    legacy_split_path: Path | None = None,
    n_splits: int = 5,
) -> EventDatasetV2:
    """读取 CSV 并构建；调用方负责指定路径，模块不包含机器绝对路径。"""

    source = pd.read_csv(source_path, low_memory=False)
    return build_event_dataset_v2(
        source,
        legacy_split_path=legacy_split_path,
        n_splits=n_splits,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_event_dataset_v2(
    dataset: EventDatasetV2,
    output_dir: Path,
    *,
    source_path: Path,
    legacy_split_path: Path | None = None,
) -> dict[str, Any]:
    """以 UTF-8 BOM CSV 写入 P1 工件并生成带 SHA-256 的构建清单。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, pd.DataFrame] = {
        "event_pair_index_v2.csv": dataset.event_pair_index,
        "as_of_admission_card_v2.csv": dataset.as_of_admission_cards,
        "evaluation_eligibility_ledger_v2.csv": dataset.evaluation_ledger,
        "outcome_ledger_v2.csv": dataset.outcome_ledger,
        "partition_manifest_v2.csv": dataset.partition_manifest,
        "cv_fold_manifest_v2.csv": dataset.cv_fold_manifest,
        "cv_fold_summary_v2.csv": dataset.cv_fold_summary,
        "count_difference_summary_v1_v2.csv": dataset.count_difference_summary,
    }
    for filename, frame in artifacts.items():
        path = output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)

    hashes = {filename: sha256_file(output_dir / filename) for filename in artifacts}
    manifest = {
        "build_id": "P1_EVENT_DATASET_V2_2024_2025",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "source": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "sample_datetime_interpretation": "recorded_at",
            "available_at_rule": "recorded_at",
            "sample_at_assumed_rule": "recorded_at - 2h",
        },
        "legacy_split": (
            {
                "path": str(legacy_split_path.resolve()),
                "sha256": sha256_file(legacy_split_path),
                "event_set_verified": dataset.legacy_split_verified,
            }
            if legacy_split_path is not None
            else None
        ),
        "counts": {
            "source_events": dataset.source_event_count,
            "event_pair_index_rows": len(dataset.event_pair_index),
            "labeled_adjacent_pairs": int(dataset.event_pair_index["target_event_id"].notna().sum()),
            "as_of_admission_rows": len(dataset.as_of_admission_cards),
            "as_of_admitted_or_warning": int(dataset.as_of_admission_cards["admission_status"].ne("REJECTED").sum()),
            "evaluation_ledger_rows": len(dataset.evaluation_ledger),
            "prospectively_sampled": int(dataset.evaluation_ledger["prospectively_sampled"].map(_as_bool).sum()),
            "nonprospective_under_2h_assumption": int(
                (~dataset.evaluation_ledger["prospectively_sampled"].map(_as_bool)).sum()
            ),
            "primary_evaluation_eligible": int(dataset.evaluation_ledger["evaluation_eligible"].map(_as_bool).sum()),
            "outcome_ledger_rows": len(dataset.outcome_ledger),
            "final_event_audit_without_label": int(dataset.event_pair_index["target_event_id"].isna().sum()),
            "cv_folds": int(dataset.cv_fold_summary["fold_id"].nunique()),
        },
        "safety": {
            "development_years_only": [2024, 2025],
            "read_2026": False,
            "llm_calls_made": 0,
            "as_of_card_future_field_check": "PASS",
            "outcomes_physically_isolated": True,
        },
        "artifacts": {
            filename: {"rows": len(artifacts[filename]), "sha256": digest} for filename, digest in hashes.items()
        },
    }
    manifest_path = output_dir / "build_manifest_v2.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest
