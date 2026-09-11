"""Persisted synthetic device: independent of historical chemistry and workbench data."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import sqlite3
import time

from copper_mvp.common import WorkbenchError, digest
from copper_mvp.control.contracts import DEVICE, POLICY, POINTS, NOTICE, ProposalInput, FaultInput

def canonical_hash(payload):
    return digest(payload)


class MockDevice:
    def __init__(self, root: Path, seed=17):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "device.sqlite"
        with self.connection() as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS device (id INTEGER PRIMARY KEY, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ledger (
                    command_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, record TEXT NOT NULL);
            """)
            c.execute("INSERT OR IGNORE INTO device VALUES(1,?)", (json.dumps(self.initial(seed)),))

    @staticmethod
    def initial(seed):
        return {"device_id": DEVICE, "environment": "MOCK", "notice": NOTICE, "policy_ref": POLICY,
                "seed": seed, "fault_script_version": "mock-faults.v1", "virtual_time": 0.,
                "device_epoch": 1, "state_version": 1, "observation_seq": 0,
                "mode": "remote", "connected": True, "running": True,
                "interlocks_ok": True, "estop_latched": False, "highest_fence": 0, "last_command_sequence": 0,
                "faults": {"reject_writes": False, "partial_write_count": 2, "ack_loss": False,
                           "stuck_pv": False, "bad_quality": False, "stale": False,
                           "readback_bias": 0., "sensor_bias": 0., "noise": 0.,
                           "ack_delay_seconds": 0., "write_delay_seconds": .2, "pv_delay_seconds": 0.},
                "points": {p: {"sp": 100., "pv": 100., "unit": "A", "quantity": "electric_current",
                           "quality": "good", "sample_time": 0., "minimum": 0., "maximum": 200.,
                           "max_delta": 10., "max_rate_per_second": 10., "response_k": .5}
                           for p in POINTS}}

    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.db_path, timeout=5)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    @staticmethod
    def _load(c):
        return json.loads(c.execute("SELECT state FROM device WHERE id=1").fetchone()[0])

    @staticmethod
    def _save(c, state):
        c.execute("UPDATE device SET state=? WHERE id=1", (json.dumps(state, allow_nan=False),))

    @staticmethod
    def _ledger(c):
        return [json.loads(r[0]) for r in c.execute("SELECT record FROM ledger ORDER BY rowid")]

    @staticmethod
    def _record(c, record):
        c.execute("UPDATE ledger SET record=? WHERE command_id=?", (json.dumps(record), record["command_id"]))

    @staticmethod
    def observed(state):
        result = copy.deepcopy(state)
        for name, point in result["points"].items():
            point["sp"] += state["faults"]["readback_bias"]
            point["pv"] += state["faults"]["sensor_bias"] + random.Random(
                f'{state["seed"]}:{state["observation_seq"]}:{name}').uniform(
                -state["faults"]["noise"], state["faults"]["noise"])
            point["quality"] = "bad" if state["faults"]["bad_quality"] or not state["connected"] else "good"
        return result

    def read_state(self):
        with self.connection() as c:
            state = self.observed(self._load(c))
            records = self._ledger(c)
        state["command_queue"] = [r["command_id"] for r in records if r["write_status"] == "pending"]
        state["execution_log"] = [{k: r[k] for k in ("command_id", "sequence", "device_epoch", "write_status")} for r in records[-20:]]
        return state

    def read_points(self):
        state = self.read_state()
        return {k: state[k] for k in ("device_id", "device_epoch", "observation_seq", "virtual_time", "points")}

    @staticmethod
    def validate(command, state):
        request = ProposalInput.model_validate(command["request"])
        if command.get("environment") != "MOCK" or command.get("policy_ref") != POLICY:
            raise WorkbenchError("仅接受合成协议命令", "MOCK_POLICY")
        if command["expected_device_epoch"] != state["device_epoch"] or command["expected_state_version"] != state["state_version"]:
            raise WorkbenchError("设备控制前提已变化，请重新提案", "STALE_PROPOSAL")
        if time.time() >= command["expires_at"]:
            raise WorkbenchError("命令已过期", "COMMAND_EXPIRED")
        if not state["connected"] or not state["running"] or state["mode"] != "remote" or state["estop_latched"] or not state["interlocks_ok"]:
            raise WorkbenchError("设备模式、连接、运行或联锁条件不允许写入", "MOCK_INTERLOCK")
        for target in request.targets:
            point = state["points"][target.point_id]
            if point["quality"] != "good" or state["virtual_time"] - point["sample_time"] > 1:
                raise WorkbenchError("观测质量或新鲜度不满足命令要求", "MOCK_QUALITY")
            if not point["minimum"] <= target.value <= point["maximum"]:
                raise WorkbenchError("设定值超出合成量程", "MOCK_RANGE")
            delta = abs(target.value - point["sp"])
            if delta > point["max_delta"] or delta / request.ramp_seconds > point["max_rate_per_second"]:
                raise WorkbenchError("设定变化量或变化率超出合成策略", "MOCK_RATE")
        return {"valid": True, "state_version": state["state_version"], "device_epoch": state["device_epoch"]}

    def validate_command(self, command):
        return self.validate(command, self.read_state())

    def submit_command(self, command, payload_hash, fencing_token, lease_until):
        if canonical_hash(command) != payload_hash:
            raise WorkbenchError("命令内容与审批哈希不一致", "PAYLOAD_CONFLICT")
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            previous = c.execute("SELECT * FROM ledger WHERE command_id=?", (command["command_id"],)).fetchone()
            if previous:
                if previous["payload_hash"] != payload_hash:
                    raise WorkbenchError("幂等键已用于其他命令", "REQUEST_CONFLICT")
                return {**json.loads(previous["record"]), "reused": True}
            state = self._load(c)
            if lease_until <= time.time() or fencing_token < state["highest_fence"]:
                raise WorkbenchError("执行器租约已失效", "STALE_FENCE")
            self.validate(command, self.observed(state))
            if any(r["write_status"] == "pending" for r in self._ledger(c)):
                raise WorkbenchError("设备仍有未完成写入", "DEVICE_BUSY")
            state["highest_fence"] = fencing_token
            state["last_command_sequence"] += 1
            now, faults = state["virtual_time"], state["faults"]
            count = 0 if faults["reject_writes"] else faults["partial_write_count"]
            accepted = command["request"]["targets"][:count]
            record = {"command_id": command["command_id"], "payload_hash": payload_hash,
                      "sequence": state["last_command_sequence"], "device_epoch": state["device_epoch"],
                      "fencing_token": fencing_token, "accepted_at": now,
                      "ack_at": now + faults["ack_delay_seconds"], "ack_lost": faults["ack_loss"],
                      "apply_at": now + faults["write_delay_seconds"], "ramp_seconds": command["request"]["ramp_seconds"],
                      "pv_after": now + faults["write_delay_seconds"] + faults["pv_delay_seconds"],
                      "accepted": accepted, "rejected": command["request"]["targets"][len(accepted):],
                      "before": {p["point_id"]: state["points"][p["point_id"]]["sp"] for p in accepted},
                      "write_status": "pending" if accepted else "rejected", "write_count": 0}
            c.execute("INSERT INTO ledger VALUES(?,?,?)", (record["command_id"], payload_hash, json.dumps(record)))
            state["state_version"] += 1
            self._save(c, state)
        return {**record, "reused": False}

    def get_command_status(self, command_id):
        with self.connection() as c:
            row = c.execute("SELECT record FROM ledger WHERE command_id=?", (command_id,)).fetchone()
            state = self._load(c)
        if not row:
            return None
        record = json.loads(row[0])
        return {**record, "ack_ready": state["virtual_time"] >= record["ack_at"]}

    def tick(self, seconds):
        if not math.isfinite(seconds) or not 0 < seconds <= 60:
            raise WorkbenchError("时钟增量应在 0 到 60 秒之间", "MOCK_CLOCK")
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            state = self._load(c)
            records = self._ledger(c)
            steps = max(1, math.ceil(seconds / .1))
            dt = seconds / steps
            for _ in range(steps):
                state["virtual_time"] += dt
                now = state["virtual_time"]
                for r in records:
                    if r["write_status"] != "pending":
                        continue
                    if not state["connected"] or not state["running"] or state["mode"] != "remote" or state["estop_latched"] or not state["interlocks_ok"]:
                        r["write_status"] = "interrupted"
                        continue
                    fraction = min(1., max(0., (now - r["apply_at"]) / r["ramp_seconds"]))
                    for target in r["accepted"]:
                        point = state["points"][target["point_id"]]
                        value = r["before"][target["point_id"]] + fraction * (target["value"] - r["before"][target["point_id"]])
                        if value != point["sp"]:
                            point["sp"] = value
                            state["state_version"] += 1
                    if fraction == 1:
                        r["write_status"], r["write_count"] = "applied", 1
                for name, point in state["points"].items():
                    delayed = any(name in r["before"] and now < r["pv_after"] for r in records)
                    if state["running"] and not state["faults"]["stuck_pv"] and not delayed:
                        # The configured response coefficient is specified per second.
                        delta = (1 - (1 - point["response_k"]) ** dt) * (point["sp"] - point["pv"])
                        point["pv"] += max(-point["max_rate_per_second"] * dt, min(point["max_rate_per_second"] * dt, delta))
            if not state["faults"]["stale"] and state["connected"]:
                state["observation_seq"] += 1
                for point in state["points"].values():
                    point["sample_time"] = state["virtual_time"]
            for record in records:
                self._record(c, record)
            self._save(c, state)
        return self.read_state()

    def inject_fault(self, fault):
        fault = FaultInput.model_validate(fault)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            state = self._load(c)
            if fault.estop_latched is False and state["estop_latched"] and not fault.reset_estop:
                raise WorkbenchError("复位急停需测试管理员明确 reset_estop", "MOCK_ESTOP")
            changes = fault.model_dump(exclude_none=True, exclude={"restart", "reset_estop"})
            for key, value in changes.items():
                if key in state:
                    state[key] = value
                else:
                    state["faults"][key] = value
            if fault.restart:
                state["device_epoch"] += 1
                state["last_command_sequence"] = 0
                for record in self._ledger(c):
                    if record["write_status"] == "pending":
                        record["write_status"] = "interrupted"
                        self._record(c, record)
            state["state_version"] += 1
            self._save(c, state)
        return self.read_state()
