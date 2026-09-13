"""Reference-based session memory rehydrated from authoritative local records."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
import time

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from copper_mvp.common import WorkbenchError, digest, dumps, utc_now, file_hash
from copper_mvp.knowledge_store import timestamp
from copper_mvp.data_contracts import source_time


class KnowledgeReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    parse_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    as_of: AwareDatetime | None = None


class MemoryCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ttl_hours: int = Field(default=168, ge=1, le=720)


class SessionMemory:
    def __init__(self, workbench, research_store):
        self.wb, self.research = workbench, research_store
        with self.research.connection() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS session_memories (
                    session_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    anchors_json TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL, expires_at TEXT NOT NULL, generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL, action TEXT NOT NULL, generation INTEGER NOT NULL, at TEXT NOT NULL);
            """)

    def selected_references(self, actor, context, *, strict=False):
        """Read only selected resources; a bookmark does not bypass current ACL."""
        refs, issues = [], []
        bound = timestamp(context["as_of"]) if context.get("as_of") else None
        for raw in context.get("knowledge_refs", []):
            item = KnowledgeReference.model_validate(raw).model_dump(mode="json")
            try:
                if bound and item["as_of"] and timestamp(item["as_of"]) > bound:
                    raise WorkbenchError("文档引用晚于任务截止时间", "AS_OF_SCOPE")
                resolved = self.wb.knowledge.resolve(actor, item["chunk_id"], item["as_of"] or bound)
                citation = resolved["citation"]
                if item["hash"] != citation["hash"] or item["parse_hash"] != citation["parse_hash"]:
                    raise WorkbenchError("文档引用版本已改变", "SOURCE_CHANGED")
                refs.append({**{k: citation[k] for k in ("doc_id", "version", "chunk_id", "hash", "parse_hash")},
                             "as_of": item["as_of"] or bound})
            except WorkbenchError as error:
                if strict:
                    raise
                issues.append({"kind": "document", "reference": item["chunk_id"], "code": error.code})
        approvals = []
        for identifier in context.get("control_command_ids", []):
            try:
                value = self.wb.control.get(actor, identifier)
                approval = value.get("approval")
                approvals.append({"command_id": identifier, "status": value["status"],
                                  "payload_hash": value["payload_hash"],
                                  "approval": {k: approval.get(k) for k in
                                               ("id", "approver_id", "decision", "revoked", "approved_at", "payload_hash")}
                                               if approval else None,
                                  "expires_at": value["payload"]["expires_at"],
                                  "expired": time.time() >= value["payload"]["expires_at"],
                                  "execution_requires_current_checks": True})
            except WorkbenchError as error:
                if strict:
                    raise
                issues.append({"kind": "command", "reference": identifier, "code": error.code})
        resource_refs, versions, constraints = {}, {}, {}
        for key in ("event_id", "optimization_run_id", "model_comparison_id", "optimizer_comparison_id"):
            identifier = context.get(key)
            if not identifier:
                continue
            try:
                if key == "event_id":
                    event = self.wb.data.row(identifier)
                    if bound and timestamp(source_time(event.decision_at)) > bound:
                        raise WorkbenchError("事件晚于任务截止", "AS_OF_SCOPE")
                elif key == "optimization_run_id":
                    value = self.wb.store.get(identifier)
                    constraints[key] = value.get("request", {})
                    versions[key] = {"fingerprint": value.get("fingerprint"),
                                     "bundle_id": value.get("request", {}).get("bundle_id"),
                                     "model_profile": value.get("request", {}).get("model_profile")}
                else:
                    if not re.fullmatch("[a-f0-9]{32}", identifier):
                        raise WorkbenchError("无效的比较引用", "RESOURCE_NOT_FOUND")
                    if key == "model_comparison_id":
                        from copper_mvp.model_comparisons import ComparisonService
                        value = ComparisonService(self.wb.root / "model_comparisons", self.wb.data).get(identifier)
                        constraints[key] = value.get("request", {})
                        versions[key] = {"fingerprint": value.get("fingerprint"), "status": value["status"]}
                        manifest = self.wb.root / "model_comparisons" / identifier / "training_manifest.json"
                        if manifest.is_file():
                            versions[key]["training_manifest_hash"] = file_hash(manifest)
                    else:
                        state_path = self.wb.root / "optimizer_comparisons" / identifier / "state.json"
                        if not state_path.is_file():
                            raise WorkbenchError("优化比较不存在", "RESOURCE_NOT_FOUND")
                        state_record = json.loads(state_path.read_text(encoding="utf-8"))
                        constraints[key] = state_record.get("request", {})
                        versions[key] = {"state_hash": file_hash(state_path)}
                resource_refs[key] = identifier
            except WorkbenchError as error:
                if strict:
                    raise
                issues.append({"kind": key, "reference": identifier, "code": error.code})
        return {"resources": resource_refs, "resource_versions": versions, "resource_constraints": constraints, "documents": refs,
                "approvals": approvals, "issues": issues, "as_of": bound}

    @staticmethod
    def _reference_anchor(references):
        return {**{k: v for k, v in references.items() if k != "resource_constraints"},
                "constraints_hash": digest(references.get("resource_constraints", {}))}

    def _authority(self, actor, session_id, current_source=None):
        actor.require("read")
        with self.research.connection() as c:
            c.execute("BEGIN")
            session = c.execute("SELECT * FROM research_sessions WHERE id=? AND owner_id=? AND project_id=?",
                                (session_id, actor.user_id, actor.project_id)).fetchone()
            if session is None:
                raise WorkbenchError("没有找到该会话", "SESSION_NOT_FOUND")
            context = json.loads(session["context_json"])
            all_tasks = c.execute("SELECT * FROM research_tasks WHERE session_id=? ORDER BY created_at DESC,id DESC",
                                  (session_id,)).fetchall()
            selected = [row for i, row in enumerate(all_tasks)
                        if i < 8 or i == len(all_tasks) - 1 or row["status"] not in ("completed", "cancelled")]
            tasks, anchors = [], []
            for row in selected:
                request = json.loads(row["request_json"])
                evidence_rows = c.execute("SELECT * FROM research_evidence WHERE task_id=? ORDER BY created_at,id",
                                          (row["id"],)).fetchall()
                evidence, evidence_refs = [], []
                for record in evidence_rows:
                    payload = json.loads(record["payload_json"])
                    expected = digest({"tool": record["tool"], "payload": payload, "source": record["source_version"]})
                    if expected != record["content_hash"]:
                        raise WorkbenchError("记忆引用的证据哈希不一致，请回查源记录", "MEMORY_EVIDENCE_HASH")
                    facts = payload.get("local_facts", {})
                    for fact in facts.values():
                        if "value" in fact and not fact.get("unit"):
                            raise WorkbenchError("本地事实缺少单位，不能构建摘要", "MEMORY_FACT_UNIT")
                    reference = {"evidence_id": record["id"].split(":", 1)[1], "hash": record["content_hash"],
                                 "source_version": record["source_version"]}
                    evidence_refs.append(reference)
                    evidence.append({**reference, "tool": record["tool"], "local_facts": facts,
                                     "summary": payload.get("summary", {})})
                settings = request.get("settings", {})
                constraints = {key: settings[key] for key in
                               ("max_cost_cny", "max_calls", "max_tool_calls", "max_wall_seconds") if key in settings}
                constraints.update(tool_effects="read_only", memory_can_authorize_actions=False,
                                   as_of=request.get("context", {}).get("as_of"))
                tasks.append({"task_id": row["id"], "status": row["status"], "version": row["version"],
                              "goal": request["question"], "context": request.get("context", {}),
                              "source_version": request["source_version"],
                              "dataset_version": request.get("dataset_version"),
                              "app_version": request.get("app_version"),
                              "source_current": current_source is None or request["source_version"] == current_source,
                              "constraints": constraints, "evidence": evidence})
                anchors.append({"task_id": row["id"], "version": row["version"],
                                "request_hash": digest({k: request.get(k) for k in
                                                        ("question", "context", "source_version", "settings", "dataset_version", "app_version")}),
                                "evidence": evidence_refs})
        for task, anchor in zip(tasks, anchors):
            task["references"] = self.selected_references(actor, task["context"])
            anchor["references"] = self._reference_anchor(task["references"])
            for evidence, reference_anchor in zip(task["evidence"], anchor["evidence"]):
                document_context = {"as_of": task["context"].get("as_of"), "knowledge_refs": []}
                for fact in evidence["local_facts"].values():
                    if fact.get("kind") == "document_reference":
                        citation = fact["citation"]
                        document_context["knowledge_refs"].append({
                            **{key: citation[key] for key in ("chunk_id", "hash", "parse_hash")},
                            "as_of": citation.get("scope_as_of", citation.get("as_of"))})
                checked = self.selected_references(actor, document_context)
                evidence["reference_issues"] = checked["issues"]
                reference_anchor["document_references"] = self._reference_anchor(checked)
                evidence["usable"] = task["source_current"] and not task["references"]["issues"] and not checked["issues"]
        selected_refs = self.selected_references(actor, context)
        initial_goal = json.loads(all_tasks[-1]["request_json"])["question"] if all_tasks else session["title"]
        live = {"session_id": session_id, "topic": session["title"], "goal": tasks[0]["goal"] if tasks else session["title"],
                "initial_goal": initial_goal,
                "selected": selected_refs, "tasks": tasks, "task_count": len(all_tasks),
                "unfinished_task_ids": [row["id"] for row in all_tasks if row["status"] not in ("completed", "cancelled")],
                "authority": "task_and_evidence_records", "memory_can_authorize_actions": False}
        stored = {"context_hash": digest(context), "tasks": anchors, "selected": self._reference_anchor(selected_refs),
                  "current_source": current_source}
        # Stored state contains identifiers, hashes and state only, never source document bodies or copied facts.
        return live, stored

    def _save(self, actor, session_id, anchors, ttl_hours, action):
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(hours=ttl_hours)).isoformat()
        with self.research.connection() as c:
            existing = c.execute("SELECT generation FROM session_memories WHERE session_id=?", (session_id,)).fetchone()
            generation = existing["generation"] + 1 if existing else 1
            c.execute("""INSERT INTO session_memories VALUES(?,?,?,?,?,?,?,?)
                         ON CONFLICT(session_id) DO UPDATE SET anchors_json=excluded.anchors_json,
                         fingerprint=excluded.fingerprint,created_at=excluded.created_at,
                         expires_at=excluded.expires_at,generation=excluded.generation""",
                      (session_id, actor.user_id, actor.project_id, dumps(anchors), digest(anchors),
                       now.isoformat(), expires, generation))
            c.execute("INSERT INTO memory_audit(session_id,actor_id,action,generation,at) VALUES(?,?,?,?,?)",
                      (session_id, actor.user_id, action, generation, utc_now()))
        return {"generation": generation, "created_at": now.isoformat(), "expires_at": expires}

    def capture(self, actor, session_id, ttl_hours=168, current_source=None):
        actor.require("compute")
        live, anchors = self._authority(actor, session_id, current_source)
        metadata = self._save(actor, session_id, anchors, ttl_hours, "capture")
        return {"status": "current", "summary": live, **metadata}

    def restore(self, actor, session_id, current_source=None, *, refresh=False):
        self.research.session(actor, session_id)
        with self.research.connection() as c:
            saved = c.execute("SELECT * FROM session_memories WHERE session_id=? AND owner_id=? AND project_id=?",
                              (session_id, actor.user_id, actor.project_id)).fetchone()
        if saved is None:
            with self.research.connection() as c:
                last = c.execute("SELECT action FROM memory_audit WHERE session_id=? ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
            if last and last["action"] == "forget":
                return {"status": "forgotten", "summary": None}
            if refresh:
                return self.capture(actor, session_id, current_source=current_source)
            return {"status": "not_captured", "summary": None}
        expired = timestamp(saved["expires_at"]) <= timestamp()
        if expired and not refresh:
            return {"status": "expired", "summary": None, "expires_at": saved["expires_at"]}
        live, anchors = self._authority(actor, session_id, current_source)
        try:
            cache_valid = digest(json.loads(saved["anchors_json"])) == saved["fingerprint"]
        except json.JSONDecodeError:
            cache_valid = False
        repaired = not cache_valid or saved["fingerprint"] != digest(anchors)
        status = "references_changed" if repaired else "current"
        metadata = {k: saved[k] for k in ("generation", "created_at", "expires_at")}
        if refresh and (expired or repaired):
            actor.require("compute")
            ttl = max(1, int((datetime.fromisoformat(saved["expires_at"]) -
                              datetime.fromisoformat(saved["created_at"])).total_seconds() / 3600))
            metadata = self._save(actor, session_id, anchors, ttl, "refresh_expired" if expired else "repair_from_authority")
            status = "refreshed"
        return {"status": status, "summary": live, "repaired_from_authority": repaired, **metadata}

    def forget(self, actor, session_id):
        self.research.session(actor, session_id)
        with self.research.connection() as c:
            c.execute("PRAGMA secure_delete=ON")
            c.execute("DELETE FROM session_memories WHERE session_id=?", (session_id,))
            c.execute("INSERT INTO memory_audit(session_id,actor_id,action,generation,at) VALUES(?,?,?,0,?)",
                      (session_id, actor.user_id, "forget", utc_now()))
        return {"forgotten": True, "session_id": session_id}

    @staticmethod
    def provider_projection(memory, excluded_task=None):
        """Only context and opaque fact references enter the existing provider path."""
        if not memory.get("summary"):
            return {}
        value = memory["summary"]
        tasks = [row for row in value["tasks"] if row["task_id"] != excluded_task][:8]
        return {"topic": value["topic"], "initial_goal": value["initial_goal"], "unfinished_task_count": len(value["unfinished_task_ids"]),
                "memory_can_authorize_actions": False, "evidence_requires_tool_refresh": True,
                "document_references_available": len(value["selected"]["documents"]),
                "unavailable_reference_count": len(value["selected"]["issues"]),
                "tasks": [{"goal": row["goal"], "status": row["status"], "source_current": row["source_current"],
                           "constraints": row["constraints"],
                           "evidence": [{"evidence_id": e["evidence_id"], "local_fact_names": list(e["local_facts"])}
                                        for e in row["evidence"] if e["usable"]]} for row in tasks]}
