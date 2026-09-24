"""Exact approvals, persistent outbox and fenced command reconciliation."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid

from copper_mvp.common import WorkbenchError, digest, dumps
from copper_mvp.control.contracts import DEVICE, NOTICE, POLICY, TERMINAL, ApprovalInput, ProposalInput
from copper_mvp.control.mock import MockDevice
from copper_mvp.control.reconcile import reconcile_feedback


class CommandService:
    def __init__(self, store, access, client=None, *, autostart=True):
        self.store, self.access, self.client = store, access, client
        self.connection = store.connection
        self.worker_id = uuid.uuid4().hex
        self.stopped = threading.Event()
        self.step_lock = threading.Lock()
        with self.connection() as c:
            if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='control_commands'").fetchone():
                backup = store.root / "migrations" / ("before_g4_" + uuid.uuid4().hex + ".sqlite")
                backup.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(backup) as target:
                    c.backup(target)
            c.executescript("""
                CREATE TABLE IF NOT EXISTS control_commands (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    request_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_hash TEXT NOT NULL, actor_auth TEXT NOT NULL,
                    status TEXT NOT NULL, approval TEXT, result TEXT NOT NULL DEFAULT '{}',
                    cancel_requested INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                    UNIQUE(project_id,owner_id,request_key));
                CREATE TABLE IF NOT EXISTS command_outbox (
                    command_id TEXT PRIMARY KEY, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS device_leases (
                    device_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL, expires_at REAL NOT NULL, command_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS command_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, command_id TEXT NOT NULL,
                    kind TEXT NOT NULL, occurred_at REAL NOT NULL, detail TEXT NOT NULL);
            """)
        self.thread = None
        if autostart and client:
            self.thread = threading.Thread(target=self._loop, name="mock-command-worker", daemon=True)
            self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=15)
        if self.client:
            self.client.close()

    def require_client(self):
        if self.client is None:
            raise WorkbenchError("独立 Mock 服务未配置", "MOCK_NOT_CONNECTED")
        return self.client

    @staticmethod
    def auth(principal):
        return {"key_hash": principal.key_hash, "user_id": principal.user_id, "role": principal.role}

    def current(self, identity, permission):
        principal = self.access.current(**identity)
        principal.require(permission)
        return principal

    @staticmethod
    def _row(c, identifier):
        row = c.execute("SELECT * FROM control_commands WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise WorkbenchError("命令不存在", "COMMAND_NOT_FOUND")
        return dict(row)

    @staticmethod
    def event(c, identifier, kind, detail):
        c.execute(
            "INSERT INTO command_events(command_id,kind,occurred_at,detail) VALUES(?,?,?,?)",
            (identifier, kind, time.time(), dumps(detail)),
        )

    def _public(self, row):
        value = {
            k: row[k]
            for k in ("id", "owner_id", "project_id", "status", "payload_hash", "cancel_requested", "created_at")
        }
        value.update(payload=json.loads(row["payload"]), result=json.loads(row["result"]), notice=NOTICE)
        approval = json.loads(row["approval"]) if row["approval"] else None
        if approval:
            approval.pop("auth", None)
        value["approval"] = approval
        with self.connection() as c:
            value["events"] = [
                dict(r) for r in c.execute("SELECT * FROM command_events WHERE command_id=? ORDER BY seq", (row["id"],))
            ]
            value["outbox"] = dict(
                c.execute("SELECT * FROM command_outbox WHERE command_id=?", (row["id"],)).fetchone() or {}
            )
        for event in value["events"]:
            event["detail"] = json.loads(event["detail"])
        return value

    def get(self, actor, identifier):
        actor.require("read")
        with self.connection() as c:
            row = self._row(c, identifier)
        if row["project_id"] != actor.project_id or (row["owner_id"] != actor.user_id and actor.role != "owner"):
            raise WorkbenchError("无权访问此命令", "FORBIDDEN")
        return self._public(row)

    def list(self, actor):
        actor.require("read")
        with self.connection() as c:
            rows = c.execute(
                (
                    "SELECT * FROM control_commands WHERE project_id=? AND (owner_id=? "
                    "OR ?='owner') ORDER BY created_at DESC LIMIT 100"
                ),
                (actor.project_id, actor.user_id, actor.role),
            ).fetchall()
        return [self._public(dict(row)) for row in rows]

    def propose(self, actor, request):
        actor.require("mock_control")
        request = ProposalInput.model_validate(request).model_dump()
        fingerprint = digest(request)
        with self.connection() as c:
            existing = c.execute(
                "SELECT * FROM control_commands WHERE project_id=? AND owner_id=? AND request_key=?",
                (actor.project_id, actor.user_id, request["request_key"]),
            ).fetchone()
        if existing:
            if existing["fingerprint"] != fingerprint:
                raise WorkbenchError("请求键已用于不同参数", "REQUEST_CONFLICT")
            return self.get(actor, existing["id"])
        state = self.require_client().read_state()
        identifier, now = uuid.uuid4().hex, time.time()
        payload = {
            "schema_version": "command.v2",
            "command_id": identifier,
            "task_id": identifier,
            "trace_id": uuid.uuid4().hex,
            "project_id": actor.project_id,
            "actor_id": actor.user_id,
            "environment": "MOCK",
            "policy_ref": POLICY,
            "device_id": DEVICE,
            "risk_level": "R3_MOCK_WRITE",
            "request": request,
            "expected_state_version": state["state_version"],
            "expected_device_epoch": state["device_epoch"],
            "created_at": now,
            "expires_at": now + request["ttl_seconds"],
            "approval_nonce": uuid.uuid4().hex,
            "preconditions": {
                "mode": "remote",
                "estop_latched": False,
                "interlocks_ok": True,
                "max_state_age_ms": 1000,
            },
            "limits": {"max_delta": 10.0, "max_rate_per_second": 10.0, "unit": "A", "minimum": 0.0, "maximum": 200.0},
            "verification": {
                "required_good_samples": 3,
                "tolerance": request["tolerance"],
                "settling_deadline_seconds": request["settling_deadline_seconds"],
                "unit": "A",
            },
            "evidence": {
                "observation_seq": state["observation_seq"],
                "virtual_time": state["virtual_time"],
                "points": state["points"],
            },
        }
        MockDevice.validate(payload, state)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute(
                "SELECT * FROM control_commands WHERE project_id=? AND owner_id=? AND request_key=?",
                (actor.project_id, actor.user_id, request["request_key"]),
            ).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise WorkbenchError("请求键已用于不同参数", "REQUEST_CONFLICT")
                identifier = existing["id"]
            else:
                c.execute(
                    """INSERT INTO control_commands(id,owner_id,project_id,request_key,fingerprint,payload,payload_hash,
                    actor_auth,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        identifier,
                        actor.user_id,
                        actor.project_id,
                        request["request_key"],
                        fingerprint,
                        dumps(payload),
                        digest(payload),
                        dumps(self.auth(actor)),
                        "WAITING_APPROVAL",
                        now,
                    ),
                )
                self.event(
                    c,
                    identifier,
                    "PROPOSED",
                    {"payload_hash": digest(payload), "state_version": state["state_version"]},
                )
        return self.get(actor, identifier)

    def approve(self, actor, identifier, decision):
        actor.require("approve")
        self.get(actor, identifier)
        decision = ApprovalInput.model_validate(decision)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = self._row(c, identifier)
            if (
                decision.payload_hash != row["payload_hash"]
                or digest(json.loads(row["payload"])) != row["payload_hash"]
            ):
                raise WorkbenchError("审批必须绑定当前完整命令", "PAYLOAD_CONFLICT")
            if row["status"] != "WAITING_APPROVAL":
                if row["approval"] and json.loads(row["approval"])["decision"] == decision.decision:
                    return self.get(actor, identifier)
                raise WorkbenchError("命令状态不允许审批", "COMMAND_STATE")
            if time.time() >= json.loads(row["payload"])["expires_at"]:
                raise WorkbenchError("提案已过期，请重新创建", "COMMAND_EXPIRED")
            approval = {
                "id": uuid.uuid4().hex,
                "approver_id": actor.user_id,
                "role": actor.role,
                "payload_hash": row["payload_hash"],
                "policy_ref": POLICY,
                "auth": self.auth(actor),
                "approved_at": time.time(),
                "revoked": False,
                "decision": decision.decision,
                "reason": decision.reason,
            }
            status = "QUEUED" if decision.decision == "approve" else "REJECTED"
            c.execute(
                "UPDATE control_commands SET approval=?,status=? WHERE id=?", (dumps(approval), status, identifier)
            )
            if status == "QUEUED":
                c.execute("INSERT INTO command_outbox(command_id,state) VALUES(?,'ready')", (identifier,))
            self.event(c, identifier, status, {"approval_id": approval["id"], "approver_id": actor.user_id})
        return self.get(actor, identifier)

    def cancel(self, actor, identifier, *, revoke=False):
        self.get(actor, identifier)
        actor.require("approve" if revoke else "mock_control")
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = self._row(c, identifier)
            if revoke and row["approval"]:
                approval = json.loads(row["approval"])
                approval["revoked"] = True
                c.execute("UPDATE control_commands SET approval=? WHERE id=?", (dumps(approval), identifier))
            if row["status"] not in TERMINAL:
                unsent = row["status"] in {"WAITING_APPROVAL", "QUEUED"}
                c.execute(
                    "UPDATE control_commands SET cancel_requested=1,status=? WHERE id=?",
                    ("CANCELLED" if unsent else row["status"], identifier),
                )
                if unsent:
                    c.execute("UPDATE command_outbox SET state='finished' WHERE command_id=?", (identifier,))
                self.event(
                    c,
                    identifier,
                    "APPROVAL_REVOKED" if revoke else "CANCEL_REQUESTED",
                    {
                        "submitted": not unsent,
                        "action": "continue_reconciliation" if not unsent else "stop_before_dispatch",
                    },
                )
        return self.get(actor, identifier)

    def _claim(self, identifier):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = self._row(c, identifier)
            if row["status"] in TERMINAL:
                return None
            lease = c.execute("SELECT * FROM device_leases WHERE device_id=?", (DEVICE,)).fetchone()
            now = time.time()
            if lease and lease["command_id"] != identifier:
                active = self._row(c, lease["command_id"])
                if active["status"] not in TERMINAL:
                    return None
            if lease and lease["worker_id"] != self.worker_id and lease["expires_at"] > now:
                return None
            token = (
                lease["fencing_token"]
                if lease
                and lease["worker_id"] == self.worker_id
                and lease["command_id"] == identifier
                and lease["expires_at"] > now
                else (lease["fencing_token"] + 1 if lease else 1)
            )
            expires = now + 10
            c.execute(
                "INSERT OR REPLACE INTO device_leases VALUES(?,?,?,?,?)",
                (DEVICE, self.worker_id, token, expires, identifier),
            )
            outbox = dict(c.execute("SELECT * FROM command_outbox WHERE command_id=?", (identifier,)).fetchone())
        return row, outbox, token, expires

    def _live(self, c, identifier, token):
        row = c.execute("SELECT * FROM device_leases WHERE device_id=?", (DEVICE,)).fetchone()
        return bool(
            row
            and row["worker_id"] == self.worker_id
            and row["command_id"] == identifier
            and row["fencing_token"] == token
            and row["expires_at"] > time.time()
        )

    def _finish_step(self, identifier, token, status, result, outbox_state):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            if not self._live(c, identifier, token):
                return
            row = self._row(c, identifier)
            if row["status"] in TERMINAL:
                return
            c.execute("UPDATE control_commands SET status=?,result=? WHERE id=?", (status, dumps(result), identifier))
            c.execute("UPDATE command_outbox SET state=? WHERE command_id=?", (outbox_state, identifier))
            if row["status"] != status:
                self.event(c, identifier, status, {"reason": result.get("reason", ""), "fencing_token": token})
            if status in TERMINAL:
                c.execute("UPDATE device_leases SET expires_at=0 WHERE device_id=?", (DEVICE,))

    def _validate_authority(self, row):
        payload = json.loads(row["payload"])
        approval = json.loads(row["approval"] or "{}")
        if digest(payload) != row["payload_hash"] or approval.get("payload_hash") != row["payload_hash"]:
            raise WorkbenchError("命令与审批哈希不一致", "PAYLOAD_CONFLICT")
        if approval.get("revoked") or approval.get("decision") != "approve":
            raise WorkbenchError("审批已撤销", "APPROVAL_REVOKED")
        self.current(json.loads(row["actor_auth"]), "mock_control")
        self.current(approval["auth"], "approve")
        return payload

    def _execute(self, identifier):
        claimed = self._claim(identifier)
        if not claimed:
            return
        row, outbox, token, lease_until = claimed
        payload = json.loads(row["payload"])
        if outbox["state"] == "ready":
            try:
                payload = self._validate_authority(row)
                state = self.require_client().read_state()
                MockDevice.validate(payload, state)
                with self.connection() as c:
                    c.execute("BEGIN IMMEDIATE")
                    current = self._row(c, identifier)
                    if current["status"] != "QUEUED" or not self._live(c, identifier, token):
                        return
                    payload = self._validate_authority(current)
                    c.execute(
                        "UPDATE command_outbox SET state='dispatching',attempts=attempts+1 WHERE command_id=?",
                        (identifier,),
                    )
                    c.execute("UPDATE control_commands SET status='DISPATCHING' WHERE id=?", (identifier,))
                    self.event(c, identifier, "DISPATCHING", {"fencing_token": token})
            except WorkbenchError as exc:
                self._finish_step(
                    identifier,
                    token,
                    "EXPIRED" if exc.code == "COMMAND_EXPIRED" else "REJECTED",
                    {"reason": str(exc), "code": exc.code},
                    "finished",
                )
                return
            try:
                receipt = self.require_client().submit_command(payload, row["payload_hash"], token, lease_until)
                result = {
                    "ack_received": receipt.get("ack_received", False),
                    "receipt": receipt,
                    "reason": "发送已返回，等待独立回读",
                }
                status = "VERIFYING"
            except WorkbenchError as exc:
                status = "UNKNOWN_OUTCOME" if exc.code == "MOCK_TRANSPORT" else "REJECTED"
                result = {"ack_received": False, "reason": str(exc), "code": exc.code}
            self._finish_step(identifier, token, status, result, "finished" if status in TERMINAL else "observing")
            return
        # A persisted dispatch intent is never replayed, including after a worker crash.
        try:
            record = self.require_client().get_command_status(identifier)
            state = self.require_client().read_state()
            status, result = reconcile_feedback(
                {**payload, "canonical_payload_hash": row["payload_hash"]}, record, state, json.loads(row["result"])
            )
        except WorkbenchError as exc:
            status, result = "UNKNOWN_OUTCOME", {**json.loads(row["result"]), "reason": str(exc), "code": exc.code}
        self._finish_step(identifier, token, status, result, "finished" if status in TERMINAL else "observing")

    def step(self):
        if not self.client:
            return
        with self.step_lock:
            with self.connection() as c:
                rows = c.execute(
                    "SELECT o.command_id FROM command_outbox o JOIN control_commands c "
                    "ON c.id=o.command_id\n"
                    "                    WHERE o.state!='finished' ORDER BY c.created_a"
                    "t"
                ).fetchall()
            for row in rows:
                self._execute(row["command_id"])

    def _loop(self):
        while not self.stopped.wait(0.2):
            try:
                self.step()
            except Exception:
                logging.exception("Mock command worker failed to advance; persisted intents retained")
                self.stopped.wait(2)
