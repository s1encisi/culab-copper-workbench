"""Authoritative sessions, task leases, evidence and shared budget accounting."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid

from copper_mvp.common import WorkbenchError, digest, dumps, utc_now

TERMINAL = {"completed", "failed", "cancelled", "needs_attention"}


class ResearchStore:
    def __init__(self, legacy_store):
        self.legacy = legacy_store
        self.root = legacy_store.root
        self.connection = legacy_store.connection
        with self.connection() as c:
            if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_sessions'").fetchone():
                backup = self.root / "migrations" / ("before_g3_" + uuid.uuid4().hex + ".sqlite")
                backup.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(backup) as destination:
                    c.backup(destination)
            c.executescript(
                "\n"
                "                CREATE TABLE IF NOT EXISTS research_sessions (\n"
                "                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, p"
                "roject_id TEXT NOT NULL,\n"
                "                    title TEXT NOT NULL, context_json TEXT NOT NUL"
                "L, created_at TEXT NOT NULL);\n"
                "                CREATE TABLE IF NOT EXISTS research_messages (\n"
                "                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_"
                "id TEXT NOT NULL, role TEXT NOT NULL,\n"
                "                    text TEXT NOT NULL, task_id TEXT, created_at T"
                "EXT NOT NULL);\n"
                "                CREATE TABLE IF NOT EXISTS research_tasks (\n"
                "                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,"
                " owner_id TEXT NOT NULL,\n"
                "                    request_key TEXT NOT NULL, fingerprint TEXT NO"
                "T NULL, request_json TEXT NOT NULL,\n"
                "                    status TEXT NOT NULL, version INTEGER NOT NULL"
                " DEFAULT 1,\n"
                "                    worker_id TEXT, fence INTEGER NOT NULL DEFAULT"
                " 0, lease_until REAL,\n"
                "                    checkpoint_json TEXT NOT NULL DEFAULT '{}', re"
                "sult_json TEXT, error_json TEXT,\n"
                "                    elapsed_ms REAL NOT NULL DEFAULT 0, model_call"
                "s INTEGER NOT NULL DEFAULT 0,\n"
                "                    tool_calls INTEGER NOT NULL DEFAULT 0, cache_h"
                "its INTEGER NOT NULL DEFAULT 0,\n"
                "                    created_at TEXT NOT NULL, started_at TEXT, fin"
                "ished_at TEXT,\n"
                "                    UNIQUE(owner_id,request_key));\n"
                "                CREATE TABLE IF NOT EXISTS research_events (\n"
                "                    seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id"
                " TEXT NOT NULL, kind TEXT NOT NULL,\n"
                "                    payload_json TEXT NOT NULL, created_at TEXT NO"
                "T NULL);\n"
                "                CREATE TABLE IF NOT EXISTS research_evidence (\n"
                "                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, to"
                "ol TEXT NOT NULL,\n"
                "                    payload_json TEXT NOT NULL, content_hash TEXT "
                "NOT NULL, source_version TEXT NOT NULL, input_hash TEXT NOT NULL,\n"
                "                    created_at TEXT NOT NULL);\n"
                "                CREATE TABLE IF NOT EXISTS research_calls (\n"
                "                    call_id TEXT PRIMARY KEY, task_id TEXT NOT NUL"
                "L, session_id TEXT NOT NULL,\n"
                "                    owner_id TEXT NOT NULL, requested_day TEXT NOT"
                " NULL, usage_json TEXT, outcome TEXT NOT NULL);\n"
                "                CREATE INDEX IF NOT EXISTS research_task_session O"
                "N research_tasks(session_id,created_at);\n"
                "                CREATE INDEX IF NOT EXISTS research_event_task ON "
                "research_events(task_id,seq);\n"
                "                CREATE INDEX IF NOT EXISTS research_evidence_task "
                "ON research_evidence(task_id);\n"
                "            "
            )

    def event(self, c, task_id, kind, payload):
        c.execute(
            "INSERT INTO research_events(task_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
            (task_id, kind, dumps(payload), utc_now()),
        )

    def create_session(self, actor, title, context):
        actor.require("read")
        identifier = uuid.uuid4().hex
        with self.connection() as c:
            c.execute(
                "INSERT INTO research_sessions VALUES(?,?,?,?,?,?)",
                (identifier, actor.user_id, actor.project_id, title, dumps(context), utc_now()),
            )
        return self.session(actor, identifier)

    def session(self, actor, session_id):
        with self.connection() as c:
            row = c.execute(
                "SELECT * FROM research_sessions WHERE id=? AND owner_id=? AND project_id=?",
                (session_id, actor.user_id, actor.project_id),
            ).fetchone()
            if row is None:
                raise WorkbenchError("没有找到该会话", "SESSION_NOT_FOUND")
            messages = c.execute(
                "SELECT role,text,task_id,created_at FROM research_messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        result = dict(row)
        result["context"] = json.loads(result.pop("context_json"))
        result["messages"] = [dict(m) for m in messages]
        return result

    def sessions(self, actor):
        with self.connection() as c:
            rows = c.execute(
                (
                    "SELECT id,title,created_at FROM research_sessions WHERE owner_id=?"
                    " AND project_id=? ORDER BY created_at DESC LIMIT 100"
                ),
                (actor.user_id, actor.project_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_task(self, actor, session_id, request):
        session = self.session(actor, session_id)
        fingerprint = digest({"session_id": session_id, "request": request})
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            old = c.execute(
                "SELECT id,fingerprint FROM research_tasks WHERE owner_id=? AND request_key=?",
                (actor.user_id, request["request_key"]),
            ).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise WorkbenchError("该请求键已用于不同问题或上下文", "REQUEST_CONFLICT")
                identifier, reused = old["id"], True
            else:
                active = c.execute(
                    (
                        "SELECT id FROM research_tasks WHERE session_id=? AND status IN ('q"
                        "ueued','running','pausing','cancelling','paused')"
                    ),
                    (session_id,),
                ).fetchone()
                if active:
                    raise WorkbenchError("请先完成、取消或恢复当前会话的任务", "SESSION_BUSY")
                identifier, reused = uuid.uuid4().hex, False
                c.execute(
                    (
                        "INSERT INTO research_tasks(id,session_id,owner_id,request_key,fing"
                        "erprint,request_json,status,created_at)\n"
                        "                             VALUES(?,?,?,?,?,?,'queued',?)"
                    ),
                    (
                        identifier,
                        session_id,
                        actor.user_id,
                        request["request_key"],
                        fingerprint,
                        dumps(request),
                        utc_now(),
                    ),
                )
                c.execute(
                    "UPDATE research_sessions SET context_json=? WHERE id=?", (dumps(request["context"]), session_id)
                )
                c.execute(
                    "INSERT INTO research_messages(session_id,role,text,task_id,created_at) VALUES(?,'user',?,?,?)",
                    (session_id, request["question"], identifier, utc_now()),
                )
                self.event(c, identifier, "queued", {"source_version": request["source_version"]})
        return self.task(actor, identifier), reused

    def internal_task(self, task_id):
        with self.connection() as c:
            row = c.execute("SELECT * FROM research_tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise WorkbenchError("没有找到该任务", "TASK_NOT_FOUND")
        result = dict(row)
        for field in ("request", "checkpoint", "result", "error"):
            result[field] = json.loads(result.pop(field + "_json") or "null")
        return result

    def task(self, actor, task_id):
        task = self.internal_task(task_id)
        self.session(actor, task["session_id"])
        task.pop("fingerprint")
        task.pop("worker_id")
        task["evidence"] = self.evidence(task_id)
        with self.connection() as c:
            task["usage"] = dict(
                c.execute(
                    """SELECT COALESCE(SUM(b.spent),0) spent_cny,
                COALESCE(SUM(CASE WHEN b.spent IS NULL THEN b.reserved ELSE 0 END),0) unsettled_cny
                FROM budget b JOIN research_calls r ON r.call_id=b.call_id WHERE r.task_id=?""",
                    (task_id,),
                ).fetchone()
            )
        return task

    def events(self, actor, task_id, after=0):
        self.session(actor, self.internal_task(task_id)["session_id"])
        with self.connection() as c:
            rows = c.execute(
                "SELECT * FROM research_events WHERE task_id=? AND seq>? ORDER BY seq LIMIT 1000", (task_id, after)
            ).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload_json"])} for r in rows]

    def claim(self, task_id, worker_id, lease_seconds=180):
        now = time.time()
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM research_tasks WHERE id=?", (task_id,)).fetchone()
            if not row or row["status"] in TERMINAL or row["status"] == "paused":
                return None
            if row["worker_id"] and (row["lease_until"] or 0) > now:
                return None
            if row["status"] not in ("queued", "running", "pausing", "cancelling"):
                return None
            fence = row["fence"] + 1
            status = row["status"] if row["status"] in ("pausing", "cancelling") else "running"
            c.execute(
                """UPDATE research_tasks SET worker_id=?,fence=?,lease_until=?,status=?,
                         version=version+1,started_at=COALESCE(started_at,?) WHERE id=?""",
                (worker_id, fence, now + lease_seconds, status, utc_now(), task_id),
            )
            self.event(c, task_id, "claimed", {"fence": fence, "recovered": row["fence"] > 0})
        return fence

    def _owned(self, c, task_id, worker_id, fence):
        row = c.execute("SELECT * FROM research_tasks WHERE id=?", (task_id,)).fetchone()
        if (
            not row
            or row["worker_id"] != worker_id
            or row["fence"] != fence
            or (row["lease_until"] or 0) <= time.time()
        ):
            raise WorkbenchError("任务租约已被其他执行器接管", "LEASE_LOST")
        return row

    def renew(self, task_id, worker_id, fence):
        with self.connection() as c:
            count = c.execute(
                """UPDATE research_tasks SET lease_until=? WHERE id=? AND worker_id=? AND fence=?
                                 AND status IN ('running','pausing','cancelling')""",
                (time.time() + 180, task_id, worker_id, fence),
            ).rowcount
        return bool(count)

    def boundary(self, task_id, worker_id, fence, checkpoint, elapsed_ms):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = self._owned(c, task_id, worker_id, fence)
            status = (
                "paused" if row["status"] == "pausing" else "cancelled" if row["status"] == "cancelling" else "running"
            )
            changed = int(status != row["status"])
            c.execute(
                "UPDATE research_tasks SET checkpoint_json=?,elapsed_ms=?,status=?,version=version+? WHERE id=?",
                (dumps(checkpoint), elapsed_ms, status, changed, task_id),
            )
            if status != "running":
                c.execute("UPDATE research_tasks SET worker_id=NULL,lease_until=NULL WHERE id=?", (task_id,))
                self.event(c, task_id, status, {})
        return status == "running"

    def control(self, actor, task_id, action, version, source_version):
        task = self.task(actor, task_id)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM research_tasks WHERE id=?", (task_id,)).fetchone()
            if row["version"] != version:
                raise WorkbenchError("任务状态已更新，请刷新后操作", "VERSION_CONFLICT")
            if action == "pause" and row["status"] in ("queued", "running"):
                status = "paused" if row["status"] == "queued" else "pausing"
            elif action == "cancel" and row["status"] not in TERMINAL:
                status = "cancelling" if row["worker_id"] and (row["lease_until"] or 0) > time.time() else "cancelled"
            elif action == "resume" and row["status"] == "paused":
                request = json.loads(row["request_json"])
                if request["source_version"] != source_version:
                    raise WorkbenchError("数据或工具版本已改变，请新建任务", "SOURCE_CHANGED")
                status = "queued"
            else:
                raise WorkbenchError("当前状态不能执行该操作", "TASK_STATE")
            c.execute("UPDATE research_tasks SET status=?,version=version+1 WHERE id=?", (status, task_id))
            self.event(c, task_id, action, {"status": status})
        return self.task(actor, task_id)

    def tool_attempt(self, task_id, worker_id, fence, maximum):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = self._owned(c, task_id, worker_id, fence)
            if row["tool_calls"] >= maximum:
                raise WorkbenchError("本次工具次数已用完", "TOOL_LIMIT")
            c.execute("UPDATE research_tasks SET tool_calls=tool_calls+1 WHERE id=?", (task_id,))

    def add_evidence(self, task_id, worker_id, fence, tool, payload, source_version, input_hash, cached=False):
        signature = digest({"tool": tool, "payload": payload, "source": source_version})
        identifier = "E-" + signature[:20]
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            self._owned(c, task_id, worker_id, fence)
            c.execute(
                "INSERT OR IGNORE INTO research_evidence VALUES(?,?,?,?,?,?,?,?)",
                (
                    task_id + ":" + identifier,
                    task_id,
                    tool,
                    dumps(payload),
                    signature,
                    source_version,
                    input_hash,
                    utc_now(),
                ),
            )
            if cached:
                c.execute("UPDATE research_tasks SET cache_hits=cache_hits+1 WHERE id=?", (task_id,))
            self.event(c, task_id, "tool_result", {"tool": tool, "evidence_id": identifier, "cached": cached})
        return {"evidence_id": identifier, "tool": tool, "data": payload, "content_hash": signature}

    def cached_evidence(self, task_id, input_hash):
        with self.connection() as c:
            row = c.execute(
                "SELECT * FROM research_evidence WHERE task_id=? AND input_hash=?", (task_id, input_hash)
            ).fetchone()
        if row:
            return {
                "evidence_id": row["id"].split(":", 1)[1],
                "tool": row["tool"],
                "data": json.loads(row["payload_json"]),
                "content_hash": row["content_hash"],
            }
        return None

    def audit(self, task_id, worker_id, fence, kind, payload):
        with self.connection() as c:
            self._owned(c, task_id, worker_id, fence)
            self.event(c, task_id, kind, payload)

    def evidence(self, task_id):
        with self.connection() as c:
            rows = c.execute(
                "SELECT * FROM research_evidence WHERE task_id=? ORDER BY created_at,id", (task_id,)
            ).fetchall()
        return [
            {
                "evidence_id": r["id"].split(":", 1)[1],
                "tool": r["tool"],
                "data": json.loads(r["payload_json"]),
                "content_hash": r["content_hash"],
                "source_version": r["source_version"],
            }
            for r in rows
        ]

    def reserve_call(self, task_id, worker_id, fence, amount, month, day, settings):
        if not math.isfinite(amount) or amount < 0:
            raise WorkbenchError("费用估计无效", "BUDGET_INPUT")
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            task = self._owned(c, task_id, worker_id, fence)
            if task["model_calls"] >= settings.max_calls:
                raise WorkbenchError("本次模型调用次数已用完", "MODEL_CALL_LIMIT")
            measure = "COALESCE(SUM(CASE WHEN b.spent IS NULL THEN b.reserved ELSE b.spent END),0)"
            global_used = c.execute(f"SELECT {measure} FROM budget b WHERE b.month=?", (month,)).fetchone()[0]
            task_used = c.execute(
                f"SELECT {measure} FROM budget b JOIN research_calls r ON r.call_id=b.call_id WHERE r.task_id=?",
                (task_id,),
            ).fetchone()[0]
            session_used = c.execute(
                f"SELECT {measure} FROM budget b JOIN research_calls r ON r.call_id=b.call_id WHERE r.session_id=?",
                (task["session_id"],),
            ).fetchone()[0]
            day_used = c.execute(
                (
                    "SELECT "
                    f"{measure}"
                    " FROM budget b JOIN research_calls r ON r.call_id=b.call_id WHERE "
                    "r.owner_id=? AND r.requested_day=?"
                ),
                (task["owner_id"], day),
            ).fetchone()[0]
            if (
                global_used + amount > settings.monthly_limit_cny
                or task_used + amount > settings.max_cost_cny
                or session_used + amount > 10
                or day_used + amount > 20
            ):
                raise WorkbenchError("任务、会话、用户或本月预算不足", "BUDGET_EXCEEDED")
            call_id = f"research:{task_id}:{task['model_calls']}"
            c.execute("INSERT INTO budget VALUES(?,?,?,?,?,?)", (call_id, month, amount, None, "reserved", utc_now()))
            c.execute(
                "INSERT INTO research_calls VALUES(?,?,?,?,?,?,?)",
                (call_id, task_id, task["session_id"], task["owner_id"], day, None, "unknown"),
            )
            c.execute("UPDATE research_tasks SET model_calls=model_calls+1 WHERE id=?", (task_id,))
            self.event(c, task_id, "model_request", {"call_id": call_id, "reserved_cny": amount})
        return call_id

    def unknown_calls(self, task_id):
        with self.connection() as c:
            return c.execute(
                "SELECT COUNT(*) FROM research_calls WHERE task_id=? AND outcome='unknown'", (task_id,)
            ).fetchone()[0]

    def settle_call(self, call_id, usage):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            c.execute(
                "UPDATE budget SET spent=?,status='settled' WHERE call_id=?", (usage["estimated_cost_cny"], call_id)
            )
            c.execute("UPDATE research_calls SET usage_json=?,outcome='known' WHERE call_id=?", (dumps(usage), call_id))

    def finish(self, task_id, worker_id, fence, result, elapsed_ms):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            task = self._owned(c, task_id, worker_id, fence)
            if task["status"] != "running":
                raise WorkbenchError("任务已请求暂停或取消", "TASK_STATE")
            c.execute(
                """UPDATE research_tasks SET status='completed',result_json=?,elapsed_ms=?,finished_at=?,
                         worker_id=NULL,lease_until=NULL,version=version+1 WHERE id=?""",
                (dumps(result), elapsed_ms, utc_now(), task_id),
            )
            c.execute(
                "INSERT INTO research_messages(session_id,role,text,task_id,created_at) VALUES(?,'assistant',?,?,?)",
                (task["session_id"], result["answer"], task_id, utc_now()),
            )
            self.event(c, task_id, "completed", {"evidence_count": len(result.get("evidence_ids", []))})

    def fail(self, task_id, worker_id, fence, code, message, elapsed_ms):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            self._owned(c, task_id, worker_id, fence)
            c.execute(
                """UPDATE research_tasks SET status='failed',error_json=?,elapsed_ms=?,finished_at=?,
                         worker_id=NULL,lease_until=NULL,version=version+1 WHERE id=?""",
                (dumps({"code": code, "message": message}), elapsed_ms, utc_now(), task_id),
            )
            if code in ("LLM_UNKNOWN_OUTCOME", "SOURCE_CHANGED"):
                c.execute("UPDATE research_tasks SET status='needs_attention' WHERE id=?", (task_id,))
            self.event(c, task_id, "failed", {"code": code})

    def pending(self):
        with self.connection() as c:
            rows = c.execute(
                """SELECT id FROM research_tasks WHERE status='queued'
                OR (status IN ('running','pausing','cancelling') AND COALESCE(lease_until,0)<?)""",
                (time.time(),),
            ).fetchall()
        return [r["id"] for r in rows]
