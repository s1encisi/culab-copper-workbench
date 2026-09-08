from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from copper_mvp.common import WorkbenchError, dumps, utc_now, write_json


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "workbench.sqlite"
        with self.connection() as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
                    fingerprint TEXT NOT NULL, task_type TEXT NOT NULL,
                    request_json TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                    result_path TEXT, error_json TEXT, duration_ms REAL);
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    node TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL, detail_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS selections (
                    run_id TEXT NOT NULL, candidate_id TEXT NOT NULL, label TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY(run_id,candidate_id));
                CREATE TABLE IF NOT EXISTS explanations (
                    cache_key TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    result_json TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS budget (
                    call_id TEXT PRIMARY KEY, month TEXT NOT NULL,
                    reserved REAL NOT NULL, spent REAL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.db_path, timeout=20)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    def create(self, request: dict, fingerprint: str) -> tuple[dict, bool]:
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute("SELECT * FROM runs WHERE request_key=?", (request["request_key"],)).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise WorkbenchError("请求键已用于不同输入，请重新创建任务", "REQUEST_CONFLICT")
                run_id = existing["run_id"]
                reused = True
            else:
                run_id = uuid.uuid4().hex
                c.execute("INSERT INTO runs(run_id,request_key,fingerprint,task_type,request_json,status,created_at) VALUES(?,?,?,?,?,?,?)", (run_id, request["request_key"], fingerprint, request["task_type"], dumps(request), "queued", utc_now()))
                reused = False
        return self.get(run_id), reused

    def directory(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise WorkbenchError("无效运行编号", "RUN_NOT_FOUND")
        path = self.root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def start(self, run_id: str):
        with self.connection() as c:
            c.execute("UPDATE runs SET status='running',started_at=? WHERE run_id=? AND status='queued'", (utc_now(), run_id))

    def trace(self, run_id: str, node: str, status: str, started_at: str, duration_ms: float, detail: dict):
        with self.connection() as c:
            c.execute("INSERT INTO traces(run_id,node,status,started_at,duration_ms,detail_json) VALUES(?,?,?,?,?,?)", (run_id, node, status, started_at, duration_ms, dumps(detail)))

    def record_result(self, run_id: str, result: dict):
        path = self.directory(run_id) / "result.json"
        write_json(path, result)
        with self.connection() as c:
            c.execute("UPDATE runs SET result_path=? WHERE run_id=?", (str(path.relative_to(self.root)), run_id))

    def finish(self, run_id: str, result: dict, duration_ms: float):
        self.record_result(run_id, result)
        with self.connection() as c:
            c.execute("UPDATE runs SET status='completed',finished_at=?,duration_ms=? WHERE run_id=?", (utc_now(), duration_ms, run_id))

    def fail(self, run_id: str, code: str, message: str, duration_ms: float = 0):
        with self.connection() as c:
            c.execute("UPDATE runs SET status='failed',finished_at=?,error_json=?,duration_ms=? WHERE run_id=?", (utc_now(), dumps({"code": code, "message": message}), duration_ms, run_id))

    def recover_interrupted(self):
        with self.connection() as c:
            c.execute("UPDATE runs SET status='failed',finished_at=?,error_json=? WHERE status IN ('queued','running')", (utc_now(), dumps({"code": "INTERRUPTED", "message": "上次进程已中断，可以重新创建任务；已保存结果仍可查看。"})))

    def get(self, run_id: str, include_result: bool = True) -> dict:
        with self.connection() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise WorkbenchError("没有找到该运行", "RUN_NOT_FOUND")
            traces = c.execute("SELECT node,status,started_at,duration_ms,detail_json FROM traces WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
            selections = c.execute("SELECT candidate_id,label,created_at FROM selections WHERE run_id=? ORDER BY created_at", (run_id,)).fetchall()
        item = dict(row)
        item["request"] = json.loads(item.pop("request_json"))
        item["error"] = json.loads(item.pop("error_json") or "null")
        item.pop("fingerprint")
        result_path = item.pop("result_path")
        item["result"] = json.loads((self.root / result_path).read_text(encoding="utf-8")) if include_result and result_path else None
        item["trace"] = [{**dict(t), "detail": json.loads(t["detail_json"])} for t in traces]
        for t in item["trace"]:
            t.pop("detail_json")
        item["selections"] = [dict(s) for s in selections]
        return item

    def list(self, task_type: str | None = None, limit: int = 100) -> list[dict]:
        with self.connection() as c:
            if task_type:
                rows = c.execute("SELECT run_id FROM runs WHERE task_type=? ORDER BY created_at DESC LIMIT ?", (task_type, limit)).fetchall()
            else:
                rows = c.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.get(r["run_id"], include_result=False) for r in rows]

    def select(self, run_id: str, candidate_id: str, label: str):
        result = self.get(run_id)["result"] or {}
        if not any(c["id"] == candidate_id for c in result.get("candidates", [])):
            raise WorkbenchError("只能保存本次结果中的候选", "CANDIDATE_NOT_FOUND")
        with self.connection() as c:
            c.execute("INSERT INTO selections VALUES(?,?,?,?) ON CONFLICT(run_id,candidate_id) DO UPDATE SET label=excluded.label", (run_id, candidate_id, label, utc_now()))

    def diagnoses(self, source_run_id: str) -> list[dict]:
        self.get(source_run_id, include_result=False)
        with self.connection() as c:
            rows = c.execute("SELECT run_id FROM runs WHERE task_type='agent_diagnostic' AND json_extract(request_json,'$.source_run_id')=? ORDER BY created_at DESC LIMIT 30", (source_run_id,)).fetchall()
        return [self.get(row["run_id"]) for row in rows]

    def cached_explanation(self, key: str) -> dict | None:
        with self.connection() as c:
            row = c.execute("SELECT result_json FROM explanations WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_explanation(self, key: str, run_id: str, value: dict):
        with self.connection() as c:
            c.execute("INSERT OR REPLACE INTO explanations VALUES(?,?,?,?)", (key, run_id, dumps(value), utc_now()))

    def reserve_cost(self, call_id: str, month: str, amount: float, monthly_limit: float):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            previous = c.execute("SELECT call_id FROM budget WHERE call_id=?", (call_id,)).fetchone()
            if previous:
                raise WorkbenchError("该说明调用已登记", "CALL_ALREADY_RESERVED")
            used = c.execute("SELECT COALESCE(SUM(CASE WHEN spent IS NULL THEN reserved ELSE spent END),0) FROM budget WHERE month=?", (month,)).fetchone()[0]
            if not math.isfinite(amount) or not math.isfinite(monthly_limit) or amount < 0 or monthly_limit <= 0 or used + amount > monthly_limit:
                raise WorkbenchError("本月说明预算不足", "BUDGET_EXCEEDED")
            c.execute("INSERT INTO budget VALUES(?,?,?,?,?,?)", (call_id, month, amount, None, "reserved", utc_now()))

    def settle_cost(self, call_id: str, spent: float):
        if not math.isfinite(spent) or spent < 0:
            raise WorkbenchError("费用必须是非负有限数", "INVALID_COST")
        with self.connection() as c:
            c.execute("UPDATE budget SET spent=?,status='settled' WHERE call_id=?", (spent, call_id))
