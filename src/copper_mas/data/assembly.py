"""合并事件层与过程快照层，形成最终预测时准入和离线建模索引。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from copper_mas.contracts.cards import AsOfAdmissionCardV2
from copper_mas.data.leakage import assert_no_future_information


PROCESS_GROUPS = ("phase_1", "phase_2", "stage34")


def _json_load(value: Any, default: Any) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    return json.loads(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def assemble_final_admission_cards(
    base_cards: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """过程覆盖是 decision_at 已知信息；缺组降级为 WARNING，不阻断基线预测。"""

    required_base = {
        "origin_event_id",
        "decision_at",
        "feature_cutoff_at",
        "admission_status",
        "reason_codes",
        "feature_group_counts",
        "missing_feature_groups",
        "source_quality_warnings",
    }
    required_coverage = {"origin_event_id", "decision_at"}
    for group in PROCESS_GROUPS:
        required_coverage.update(
            {
                f"{group}__unique_matched_record_count",
                f"{group}__group_4_of_7_ready",
            }
        )
    if missing := required_base - set(base_cards.columns):
        raise ValueError(f"基础准入卡缺列: {sorted(missing)}")
    if missing := required_coverage - set(coverage.columns):
        raise ValueError(f"过程覆盖表缺列: {sorted(missing)}")

    base = base_cards.copy()
    cov = coverage.copy()
    base["decision_at"] = pd.to_datetime(base["decision_at"], errors="raise")
    base["feature_cutoff_at"] = pd.to_datetime(base["feature_cutoff_at"], errors="raise")
    cov["decision_at"] = pd.to_datetime(cov["decision_at"], errors="raise")
    merged = base.merge(
        cov,
        on="origin_event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_coverage"),
    )
    if merged["decision_at_coverage"].isna().any():
        raise ValueError("部分起点缺少过程覆盖记录")
    if not merged["decision_at"].eq(merged["decision_at_coverage"]).all():
        raise ValueError("基础准入卡与过程覆盖的 decision_at 不一致")

    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        counts = dict(_json_load(row["feature_group_counts"], {}))
        missing_groups = list(_json_load(row["missing_feature_groups"], []))
        warnings = list(_json_load(row["source_quality_warnings"], []))
        reasons = list(_json_load(row["reason_codes"], []))
        for group in PROCESS_GROUPS:
            count = int(row[f"{group}__unique_matched_record_count"])
            ready = _as_bool(row[f"{group}__group_4_of_7_ready"])
            counts[group] = count
            if not ready:
                missing_groups.append(group)
                warnings.append(f"CORE_PROCESS_GROUP_LT_4_OF_7:{group}")

        base_status = str(row["admission_status"])
        if base_status == "REJECTED":
            status = "REJECTED"
        elif missing_groups or warnings:
            status = "WARNING"
        else:
            status = "ADMITTED"

        card = AsOfAdmissionCardV2(
            origin_event_id=str(row["origin_event_id"]),
            decision_at=row["decision_at"].to_pydatetime(),
            feature_cutoff_at=row["feature_cutoff_at"].to_pydatetime(),
            admission_status=status,
            reason_codes=tuple(sorted(set(reasons))),
            feature_group_counts=counts,
            missing_feature_groups=tuple(sorted(set(missing_groups))),
            source_quality_warnings=tuple(sorted(set(warnings))),
        )
        assert_no_future_information(card, decision_at=card.decision_at)
        payload = card.model_dump(mode="json")
        for field in (
            "reason_codes",
            "feature_group_counts",
            "missing_feature_groups",
            "source_quality_warnings",
        ):
            payload[field] = json.dumps(
                payload[field], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        payload["core_process_4_of_7_ready"] = not missing_groups
        rows.append(payload)
    return pd.DataFrame(rows)


def build_training_evaluation_index(
    pair_index: pd.DataFrame,
    final_cards: pd.DataFrame,
    evaluation_ledger: pd.DataFrame,
    outcome_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """该表只供离线训练装配；绝不能作为 ForecastRequest。"""

    pairs = pair_index[
        ["pair_id", "origin_event_id", "origin_recorded_at", "origin_year", "pair_status"]
    ].copy()
    cards = final_cards[
        ["origin_event_id", "admission_status", "core_process_4_of_7_ready"]
    ].copy()
    evaluations = evaluation_ledger[
        ["pair_id", "prospectively_sampled", "evaluation_eligible"]
    ].copy()
    outcome_pairs = set(outcome_ledger["pair_id"].astype(str))
    result = pairs.merge(cards, on="origin_event_id", how="left", validate="many_to_one")
    result = result.merge(evaluations, on="pair_id", how="left", validate="one_to_one")
    result["outcome_present"] = result["pair_id"].astype(str).isin(outcome_pairs)
    result["runtime_prediction_available"] = result["admission_status"].ne("REJECTED")
    result["tier0_labeled_development"] = (
        result["runtime_prediction_available"] & result["outcome_present"]
    )
    result["tier0_primary_prospective"] = (
        result["runtime_prediction_available"]
        & result["evaluation_eligible"].map(_as_bool)
    )
    result["tier1_core_labeled_development"] = (
        result["tier0_labeled_development"]
        & result["core_process_4_of_7_ready"].map(_as_bool)
    )
    result["tier1_core_primary_prospective"] = (
        result["tier0_primary_prospective"]
        & result["core_process_4_of_7_ready"].map(_as_bool)
    )
    return result
