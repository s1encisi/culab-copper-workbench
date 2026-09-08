"""只向后的多锚点 as-of 快照与统计特征。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_anchor_grid(
    events: pd.DataFrame,
    *,
    event_id_column: str,
    decision_column: str,
    anchor_offsets_hours: Iterable[int],
) -> pd.DataFrame:
    required = {event_id_column, decision_column}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"事件表缺列: {sorted(missing)}")
    base = events[[event_id_column, decision_column]].copy()
    base[decision_column] = pd.to_datetime(base[decision_column], errors="raise")
    offsets = pd.DataFrame({"anchor_offset_hours": list(anchor_offsets_hours)})
    base["_join_key"] = 1
    offsets["_join_key"] = 1
    anchors = base.merge(offsets, on="_join_key", how="inner").drop(columns="_join_key")
    anchors["anchor_at"] = anchors[decision_column] - pd.to_timedelta(
        anchors["anchor_offset_hours"], unit="h"
    )
    return anchors.sort_values(["anchor_at", event_id_column, "anchor_offset_hours"]).reset_index(drop=True)


def backward_asof_snapshots(
    events: pd.DataFrame,
    records: pd.DataFrame,
    *,
    event_id_column: str = "origin_event_id",
    decision_column: str = "decision_at",
    record_time_column: str = "timestamp",
    record_id_column: str,
    anchor_offsets_hours: Iterable[int] = (0, 2, 4, 6, 8, 10, 12),
    tolerance_hours: float = 2.0,
) -> pd.DataFrame:
    """每个锚点只匹配该锚点之前最近的记录，绝不使用未来最近值。"""

    if tolerance_hours <= 0:
        raise ValueError("tolerance_hours 必须大于 0")
    if record_time_column not in records or record_id_column not in records:
        raise ValueError("过程表缺少时间列或事件ID列")

    anchors = build_anchor_grid(
        events,
        event_id_column=event_id_column,
        decision_column=decision_column,
        anchor_offsets_hours=anchor_offsets_hours,
    )
    right = records.copy()
    right[record_time_column] = pd.to_datetime(right[record_time_column], errors="coerce")
    right = right[right[record_time_column].notna()].sort_values(record_time_column).reset_index(drop=True)

    merged = pd.merge_asof(
        anchors.sort_values("anchor_at"),
        right,
        left_on="anchor_at",
        right_on=record_time_column,
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
        allow_exact_matches=True,
    )
    merged = merged.rename(
        columns={record_time_column: "matched_at", record_id_column: "matched_record_id"}
    )
    merged["available_at"] = merged["matched_at"]
    merged["anchor_match_age_hours"] = (
        merged["anchor_at"] - merged["matched_at"]
    ).dt.total_seconds() / 3600
    merged["observation_age_at_decision_hours"] = (
        merged[decision_column] - merged["matched_at"]
    ).dt.total_seconds() / 3600
    merged["missing_record"] = merged["matched_record_id"].isna()

    bad = merged["matched_at"].notna() & (
        (merged["matched_at"] > merged["anchor_at"])
        | (merged["available_at"] > merged[decision_column])
    )
    if bad.any():
        raise AssertionError("as-of 快照出现未来记录")
    return merged.sort_values(
        [event_id_column, "anchor_offset_hours"]
    ).reset_index(drop=True)


def summarize_feature_anchors(
    snapshots: pd.DataFrame,
    *,
    event_id_column: str,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """把七个锚点展开为宽表，并计算仅依赖历史快照的12h统计。"""

    features = list(feature_columns)
    event_index = pd.Index(
        snapshots[event_id_column].drop_duplicates(), name=event_id_column
    )
    columns: dict[str, pd.Series] = {}
    for feature in features:
        pivot = snapshots.pivot(index=event_id_column, columns="anchor_offset_hours", values=feature)
        age_pivot = snapshots.pivot(
            index=event_id_column,
            columns="anchor_offset_hours",
            values="observation_age_at_decision_hours",
        )
        for offset in sorted(pivot.columns):
            suffix = f"t_minus_{int(offset)}h"
            columns[f"{feature}__{suffix}"] = pivot[offset].reindex(event_index)
            columns[f"{feature}__{suffix}__missing"] = (
                pivot[offset].isna().astype("int8").reindex(event_index)
            )
            columns[f"{feature}__{suffix}__age_h"] = (
                age_pivot[offset].where(pivot[offset].notna()).reindex(event_index)
            )

        ordered_offsets = np.array(sorted(pivot.columns), dtype=float)
        values = pivot.reindex(columns=ordered_offsets.astype(int))
        columns[f"{feature}__mean_12h"] = values.mean(axis=1, skipna=True).reindex(event_index)
        columns[f"{feature}__std_12h"] = values.std(
            axis=1, skipna=True, ddof=0
        ).reindex(event_index)
        columns[f"{feature}__range_12h"] = (
            values.max(axis=1, skipna=True) - values.min(axis=1, skipna=True)
        ).reindex(event_index)
        columns[f"{feature}__valid_count_12h"] = (
            values.notna().sum(axis=1).astype("int8").reindex(event_index)
        )

        chronological_x = -ordered_offsets

        def slope(row: pd.Series) -> float:
            mask = row.notna().to_numpy()
            if mask.sum() < 2:
                return np.nan
            return float(np.polyfit(chronological_x[mask], row.to_numpy(dtype=float)[mask], 1)[0])

        columns[f"{feature}__slope_per_h_12h"] = values.apply(slope, axis=1).reindex(
            event_index
        )
    return pd.DataFrame(columns, index=event_index).reset_index()
