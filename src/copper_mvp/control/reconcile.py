"""Feedback requires device-ledger attribution plus SP and three fresh PV samples."""
from __future__ import annotations

import copy


def reconcile_feedback(command, record, state, previous):
    result = copy.deepcopy(previous or {})
    result["environment"] = "MOCK"
    result["observation_seq"] = state["observation_seq"]
    result["observed_device_epoch"] = state["device_epoch"]
    if record is None:
        result["reason"] = "设备账本尚无该命令，保留未知结果，未重发"
        return "UNKNOWN_OUTCOME", result
    result["receipt"] = record
    if record["device_epoch"] != command["expected_device_epoch"] or state["device_epoch"] != record["device_epoch"]:
        result["reason"] = "设备启动 epoch 已变化，需要人工核对原命令与当前状态"
        return "UNKNOWN_OUTCOME", result
    if record["payload_hash"] != command["canonical_payload_hash"]:
        result["reason"] = "设备账本的命令哈希不匹配"
        return "UNKNOWN_OUTCOME", result
    result["ack_confirmed_by_status"] = bool(record.get("ack_ready") and not record["ack_lost"])
    if record["write_status"] == "rejected":
        result["reason"] = "设备拒绝写入"
        result["points"] = {p["point_id"]: {"status": "rejected", "target": p["value"]} for p in record["rejected"]}
        return "REJECTED", result
    if record["write_status"] == "interrupted":
        result["reason"] = "设备写入期间控制前提改变；已产生的变化保留在回读证据"
        result["readback"] = state["points"]
        return "FAILED", result
    old_seq = (previous or {}).get("observation_seq", -1)
    fresh_sample = state["observation_seq"] > old_seq
    points = result.setdefault("points", {})
    ready = []
    for target in record["accepted"]:
        name = target["point_id"]
        point = state["points"][name]
        before = points.get(name, {})
        good = (point["quality"] == "good" and state["virtual_time"] - point["sample_time"] <= 1
                and abs(point["sp"] - target["value"]) < 1e-8
                and abs(point["pv"] - target["value"]) <= command["request"]["tolerance"])
        count = (before.get("good_samples", 0) + 1 if good else 0) if fresh_sample else before.get("good_samples", 0)
        points[name] = {"target": target["value"], "sp": point["sp"], "pv": point["pv"], "unit": point["unit"],
                        "quality": point["quality"], "sample_time": point["sample_time"],
                        "good_samples": count, "status": "settled" if count >= 3 and good else "tracking"}
        ready.append(count >= 3 and good)
    for target in record["rejected"]:
        points[target["point_id"]] = {"target": target["value"], "status": "rejected",
                                     "readback": state["points"][target["point_id"]]}
    deadline = record["accepted_at"] + command["request"]["settling_deadline_seconds"]
    result["virtual_deadline"] = deadline
    result["virtual_time"] = state["virtual_time"]
    if fresh_sample:
        result.setdefault("samples", []).append({
            "observation_seq": state["observation_seq"], "time": state["virtual_time"],
            "points": copy.deepcopy(points)})
    if ready and all(ready) and state["virtual_time"] <= deadline:
        result["reason"] = "已写入的点通过连续三次合格回读"
        return ("PARTIAL" if record["rejected"] else "VERIFIED"), result
    if state["virtual_time"] >= deadline:
        result["reason"] = "回读截止时间内未满足全部稳定条件"
        return ("PARTIAL" if record["rejected"] else "FAILED"), result
    result["reason"] = "等待新的 SP/PV 观测"
    return "VERIFYING", result
