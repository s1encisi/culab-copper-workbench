"""把30条人工抽核结果转换成可审计的释放条件。"""

from __future__ import annotations

from typing import Any

import pandas as pd


TIME_COLUMN = "人工核对：时间和相邻关系是否合理"
OPERATION_COLUMN = "人工核对：是否存在表外切槽/出铜/其他操作"
MODE_COLUMN = "人工核对：工况判断是否合理"
NOTE_COLUMN = "人工备注"
REQUIRED_ROWS = 30
ALLOWED = {
    TIME_COLUMN: {"是", "否", "不确定"},
    OPERATION_COLUMN: {"有", "无", "不确定"},
    MODE_COLUMN: {"是", "否", "不确定"},
}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def assess_manual_review(frame: pd.DataFrame) -> dict[str, Any]:
    missing_columns = sorted(set(ALLOWED) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"人工抽核表缺少填写列: {missing_columns}")
    if len(frame) != REQUIRED_ROWS:
        raise ValueError(f"人工抽核必须保持30条；实际{len(frame)}条")

    invalid: list[dict[str, Any]] = []
    blank: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for column, allowed in ALLOWED.items():
        values = frame[column].map(_text)
        counts[column] = {option: int((values == option).sum()) for option in sorted(allowed)}
        for index, value in values.items():
            sequence = int(frame.loc[index, "抽核序号"]) if "抽核序号" in frame else int(index) + 1
            if not value:
                blank.append({"抽核序号": sequence, "字段": column})
            elif value not in allowed:
                invalid.append({"抽核序号": sequence, "字段": column, "值": value})

    completed = not blank and not invalid
    time_failures = counts[TIME_COLUMN].get("否", 0) + counts[TIME_COLUMN].get("不确定", 0)
    mode_failures = counts[MODE_COLUMN].get("否", 0) + counts[MODE_COLUMN].get("不确定", 0)
    operation_flags = counts[OPERATION_COLUMN].get("有", 0) + counts[OPERATION_COLUMN].get("不确定", 0)
    accepted = completed and time_failures == 0 and mode_failures == 0 and operation_flags == 0
    if not completed:
        status = "INCOMPLETE"
    elif accepted:
        status = "COMPLETED_AND_ACCEPTED"
    else:
        status = "COMPLETED_REQUIRES_ADJUDICATION"
    return {
        "review_id": "P1_MANUAL_REVIEW_30_V2",
        "row_count": len(frame),
        "completion_status": status,
        "completed": completed,
        "accepted_for_external_release": accepted,
        "counts": counts,
        "blank_cells": blank,
        "invalid_cells": invalid,
        "adjudication_counts": {
            "time_or_adjacency_negative_or_uncertain": time_failures,
            "mode_negative_or_uncertain": mode_failures,
            "external_operation_present_or_uncertain": operation_flags,
        },
        "rule": (
            "只有30条均完成，且时间=是、表外操作=无、工况=是时，"
            "才自动达到COMPLETED_AND_ACCEPTED；其他情况需逐条裁决。"
        ),
    }
