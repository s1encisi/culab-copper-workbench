from __future__ import annotations

import math
from datetime import datetime

from copper_mvp.common import WorkbenchError


def verify_prediction(result: dict, context: dict) -> dict:
    if result["event_id"] != context["event_id"] or result["decision_at"] != context["decision_at"]:
        raise WorkbenchError("预测的事件或时间不一致", "PREDICTION_IDENTITY")
    if result["dataset_version"] != context["dataset_version"]:
        raise WorkbenchError("预测数据版本不一致", "DATA_VERSION")
    if result["model_scope"] == "oof_replay" and datetime.fromisoformat(result["fit_cutoff_at"]) > datetime.fromisoformat(context["decision_at"]):
        raise WorkbenchError("模型拟合截止时间晚于事件", "FUTURE_MODEL")
    for target, unit in (("cu", "g/L"), ("as", "mg/L")):
        p = result["predictions"][target]
        if p["value"] is None:
            if context[target] is not None:
                raise WorkbenchError("可用目标未生成结果", "MISSING_PREDICTION")
            continue
        if p["unit"] != unit or p["current"] != context[target] or not math.isfinite(p["value"]):
            raise WorkbenchError("预测数值、当前值或单位不一致", "PREDICTION_VALUES")
        if not math.isclose(p["value"] - p["current"], p["delta"], rel_tol=1e-9, abs_tol=1e-8):
            raise WorkbenchError("预测变化量不一致", "PREDICTION_DELTA")
        if p.get("interval") is not None:
            low, high = p["interval"]
            if not all(math.isfinite(v) for v in (low, high)) or not low <= p["value"] <= high:
                raise WorkbenchError("预测区间不包含点预测", "INVALID_INTERVAL")
    return {"passed": True, "checks": ["事件与时间一致", "数据和模型版本一致", "模型截止时间正确", "数值、单位与变化量一致"]}
