"""按预注册规则合并 P2/P2.1 证据并应用可复现 fallback。"""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_PROTOCOL_KEYS = {
    "protocol_id",
    "targets",
    "candidate_allowlist",
    "reference_model",
    "promotion_thresholds",
    "fallback_rule",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def evaluate_p2_v1_candidates(
    fold_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    """把早期绝对目标基线转换为同一冻结门槛证据。"""

    missing_keys = REQUIRED_PROTOCOL_KEYS - set(protocol)
    if missing_keys:
        raise ValueError(f"选模协议缺少字段: {sorted(missing_keys)}")
    required_fold = {"target_name", "model_name", "fold_id", "mae", "persistence_mae"}
    required_overall = {
        "target_name",
        "model_name",
        "pooled_mae",
        "pooled_rmse",
        "pooled_persistence_mae",
        "pooled_persistence_rmse",
    }
    if required_fold - set(fold_metrics):
        raise ValueError("P2 fold metrics 字段不完整")
    if required_overall - set(overall_metrics):
        raise ValueError("P2 overall metrics 字段不完整")
    thresholds = protocol["promotion_thresholds"]
    allowlist = set(protocol["candidate_allowlist"])
    reference = protocol["reference_model"]
    rows: list[dict[str, Any]] = []
    for record in overall_metrics.to_dict(orient="records"):
        model = str(record["model_name"])
        target = str(record["target_name"])
        if model not in allowlist or target not in set(protocol["targets"]):
            continue
        folds = fold_metrics.loc[
            (fold_metrics["model_name"].astype(str) == model) & (fold_metrics["target_name"].astype(str) == target)
        ].copy()
        mae_ratios = pd.to_numeric(folds["mae"]) / pd.to_numeric(folds["persistence_mae"])
        pooled_mae_ratio = float(record["pooled_mae"]) / float(record["pooled_persistence_mae"])
        pooled_rmse_ratio = float(record["pooled_rmse"]) / float(record["pooled_persistence_rmse"])
        is_reference = model == reference
        gates = {
            "gate_pooled_mae_passed": pooled_mae_ratio <= float(thresholds["pooled_mae_ratio_vs_persistence_max"]),
            "gate_pooled_rmse_passed": pooled_rmse_ratio <= float(thresholds["pooled_rmse_ratio_vs_persistence_max"]),
            "gate_fold_wins_passed": int((mae_ratios < 1).sum()) >= int(thresholds["folds_with_mae_improvement_min"]),
            "gate_worst_fold_passed": float(mae_ratios.max())
            <= float(thresholds["worst_fold_mae_ratio_vs_persistence_max"]),
        }
        eligible = (not is_reference) and all(gates.values())
        rows.append(
            {
                "protocol_id": protocol["protocol_id"],
                "source_experiment": "P2_BASELINES_V1",
                "target_name": target,
                "model_name": model,
                "is_reference_model": is_reference,
                "hard_gates_passed": True,
                "pooled_mae_ratio_vs_persistence": pooled_mae_ratio,
                "pooled_rmse_ratio_vs_persistence": pooled_rmse_ratio,
                "folds_with_mae_improvement": int((mae_ratios < 1).sum()),
                "worst_fold_mae_ratio_vs_persistence": float(mae_ratios.max()),
                "median_fold_mae_ratio_vs_persistence": float(mae_ratios.median()),
                "fold_mae_ratio_standard_deviation": float(mae_ratios.std(ddof=1)),
                **gates,
                "eligible_for_promotion": eligible,
                "eligibility_status": (
                    "REFERENCE_MODEL" if is_reference else "PASSES_FROZEN_GATES" if eligible else "FAILS_FROZEN_GATES"
                ),
            }
        )
    return pd.DataFrame(rows)


def normalize_p2_1_eligibility(evidence: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    required = {
        "target_name",
        "model_name",
        "is_reference_model",
        "hard_gates_passed",
        "pooled_mae_ratio_vs_persistence",
        "pooled_rmse_ratio_vs_persistence",
        "folds_with_mae_improvement",
        "worst_fold_mae_ratio_vs_persistence",
        "median_fold_mae_ratio_vs_persistence",
        "fold_mae_ratio_standard_deviation",
        "gate_pooled_mae_passed",
        "gate_pooled_rmse_passed",
        "gate_fold_wins_passed",
        "gate_worst_fold_passed",
        "eligible_for_promotion",
        "eligibility_status",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"P2.1 eligibility 字段缺失: {sorted(missing)}")
    normalized = evidence.loc[:, sorted(required)].copy()
    normalized.insert(0, "source_experiment", "P2_1_RESIDUAL_BASELINES_V1")
    normalized.insert(0, "protocol_id", protocol["protocol_id"])
    for column in (
        "is_reference_model",
        "hard_gates_passed",
        "gate_pooled_mae_passed",
        "gate_pooled_rmse_passed",
        "gate_fold_wins_passed",
        "gate_worst_fold_passed",
        "eligible_for_promotion",
    ):
        normalized[column] = normalized[column].map(_as_bool)
    return normalized


def select_per_target(evidence: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """通过全部门槛才晋级；无人晋级时按协议回退 Persistence。"""

    if protocol["fallback_rule"] != "IF_NO_CANDIDATE_PASSES_ALL_GATES_SELECT_PERSISTENCE":
        raise ValueError("未知 fallback_rule")
    complexity = protocol.get("complexity_rank") or {}
    selected: dict[str, dict[str, Any]] = {}
    for target in protocol["targets"]:
        candidates = evidence.loc[
            (evidence["target_name"].astype(str) == target) & evidence["eligible_for_promotion"].map(_as_bool)
        ].copy()
        if candidates.empty:
            selected[target] = {
                "model_name": protocol["reference_model"],
                "selection_reason": "FALLBACK_NO_CANDIDATE_PASSED_ALL_FROZEN_GATES",
            }
            continue
        candidates["complexity_rank"] = candidates["model_name"].map(complexity)
        candidates = candidates.sort_values(
            [
                "median_fold_mae_ratio_vs_persistence",
                "pooled_mae_ratio_vs_persistence",
                "fold_mae_ratio_standard_deviation",
                "complexity_rank",
            ],
            kind="mergesort",
        )
        winner = candidates.iloc[0]
        selected[target] = {
            "model_name": str(winner["model_name"]),
            "selection_reason": "PASSED_ALL_FROZEN_GATES_AND_RANKED_FIRST",
        }
    return selected
