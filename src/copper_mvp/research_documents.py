"""Exact document disclosure consent and source-linked research answer lifecycle."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import uuid

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from typing import Literal

from copper_mvp.access import Principal
from copper_mvp.common import WorkbenchError, dumps, utc_now
from copper_mvp.knowledge_parsing import token_count
from copper_mvp.knowledge_store import timestamp

PROVIDER = "deepseek-flash"
REDACTED = "关联文档的授权已失效，原回答内容已清除。"


class DisclosureConsent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    parse_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider: Literal["deepseek-flash"] = PROVIDER
    max_excerpt_tokens: int = Field(default=1200, ge=64, le=2000)
    expires_at: AwareDatetime
    reason: str = Field(min_length=1, max_length=500)


class ResearchDocuments:
    def __init__(self, workbench, research):
        self.wb, self.research = workbench, research
        self.knowledge = workbench.knowledge
        with self.knowledge.connection() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS document_disclosures (
                    id TEXT PRIMARY KEY,doc_id TEXT NOT NULL,version INTEGER NOT NULL,
                    document_hash TEXT NOT NULL,parse_hash TEXT NOT NULL,provider TEXT NOT NULL,
                    max_excerpt_tokens INTEGER NOT NULL,expires_at TEXT NOT NULL,
                    approver_id TEXT NOT NULL,approver_role TEXT NOT NULL,auth_ref TEXT NOT NULL,
                    reason TEXT NOT NULL,created_at TEXT NOT NULL,revoked INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(doc_id,version) REFERENCES document_versions(doc_id,version) ON DELETE CASCADE);
            """)
        with self.research.connection() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS research_document_sources (
                    task_id TEXT NOT NULL,doc_id TEXT NOT NULL,version INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,approval_id TEXT NOT NULL,reference_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,PRIMARY KEY(task_id,chunk_id,approval_id));
                CREATE INDEX IF NOT EXISTS research_sources_document ON research_document_sources(doc_id);
                CREATE TABLE IF NOT EXISTS research_document_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,action TEXT NOT NULL,at TEXT NOT NULL);
            """)
        self.knowledge.change_listeners.append(self.on_document_change)
        with self.research.connection() as c:
            recorded_documents = [row[0] for row in c.execute("SELECT DISTINCT doc_id FROM research_document_sources")]
        for doc_id in recorded_documents:
            self.invalidate(doc_id)

    def grant(self, actor, doc_id, version, consent: DisclosureConsent):
        actor.require("approve")
        now = datetime.now(timezone.utc)
        if not now < consent.expires_at <= now + timedelta(days=30):
            raise WorkbenchError("正文授权有效期需在未来 30 天内", "DOCUMENT_CONSENT_TIME")
        self.wb.access.current(actor.key_hash, actor.user_id, actor.role)
        with self.knowledge.lock, self.knowledge.connection() as c:
            document = self.knowledge._document(c, actor, doc_id, manage=True)
            if document["revoked"]:
                raise WorkbenchError("文档已撤销，不能授予正文权限", "DOCUMENT_SOURCE_REVOKED")
            row = self.knowledge._version(c, doc_id, version)
            if (row["status"] != "indexed" or row["content_hash"] != consent.document_hash
                    or row["index_parse_hash"] != consent.parse_hash):
                raise WorkbenchError("授权必须绑定已审核索引的准确文档和解析版本", "VERSION_CONFLICT")
            identifier = uuid.uuid4().hex
            c.execute("INSERT INTO document_disclosures VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                      (identifier,doc_id,version,consent.document_hash,consent.parse_hash,consent.provider,
                       consent.max_excerpt_tokens,timestamp(consent.expires_at),actor.user_id,actor.role,
                       actor.key_hash,consent.reason,utc_now()))
            self.knowledge._audit(c,actor,doc_id,"grant_model_excerpt")
        return {"id":identifier,"doc_id":doc_id,"version":version,
                **consent.model_dump(mode="json"),"revoked":False}

    def consents(self, actor, doc_id, version):
        with self.knowledge.lock, self.knowledge.connection() as c:
            self.knowledge._document(c,actor,doc_id,manage=True)
            rows=c.execute("SELECT * FROM document_disclosures WHERE doc_id=? AND version=? ORDER BY created_at",
                           (doc_id,version)).fetchall()
        return [{k:row[k] for k in row.keys() if k!="auth_ref"} for row in rows]

    def revoke(self, actor, doc_id, consent_id):
        actor.require("approve")
        with self.knowledge.lock:
            with self.knowledge.connection() as c:
                self.knowledge._document(c,actor,doc_id,manage=True)
                if not c.execute("SELECT 1 FROM document_disclosures WHERE id=? AND doc_id=?",
                                 (consent_id,doc_id)).fetchone():
                    raise WorkbenchError("没有该正文授权", "DOCUMENT_CONSENT_NOT_FOUND")
                c.execute("UPDATE document_disclosures SET revoked=1 WHERE id=?", (consent_id,))
                self.knowledge._audit(c,actor,doc_id,"revoke_model_excerpt")
            self.invalidate(doc_id, approval_id=consent_id)
        return {"id":consent_id,"revoked":True}

    def _read(self, actor, reference):
        scope=reference.get("scope_as_of",reference.get("as_of"))
        resolved=self.knowledge.resolve(actor,reference["chunk_id"],scope)
        actual=resolved["citation"]
        if (actual["hash"] != reference["hash"] or actual["parse_hash"] != reference["parse_hash"]
                or actual["document_hash"] != reference.get("document_hash",actual["document_hash"])):
            raise WorkbenchError("文档内容或解析版本已改变", "DOCUMENT_SOURCE_CHANGED")
        return resolved

    def _consent(self, actor, reference, approval_id=None):
        resolved=self._read(actor,reference)
        citation=resolved["citation"]
        with self.knowledge.connection() as c:
            rows=c.execute("""SELECT * FROM document_disclosures WHERE doc_id=? AND version=? AND document_hash=?
                              AND parse_hash=? AND provider=? AND revoked=0 AND expires_at>? ORDER BY created_at DESC""",
                           (citation["doc_id"],citation["version"],citation["document_hash"],
                            citation["parse_hash"],PROVIDER,timestamp())).fetchall()
        for row in rows:
            if approval_id is not None and row["id"]!=approval_id:
                continue
            try:
                self.wb.access.current(row["auth_ref"],row["approver_id"],row["approver_role"])
            except WorkbenchError:
                continue
            if token_count(resolved["text"])<=row["max_excerpt_tokens"]:
                return dict(row),resolved
        return None,resolved

    def sources(self, task_id):
        with self.research.connection() as c:
            return [dict(row) for row in c.execute("SELECT * FROM research_document_sources WHERE task_id=? ORDER BY created_at,chunk_id",
                                                  (task_id,))]

    def assert_valid(self, actor, task_id, *, require_consent=True):
        for source in self.sources(task_id):
            reference=json.loads(source["reference_json"])
            if require_consent:
                approval,_=self._consent(actor,reference,source["approval_id"])
                if approval is None:
                    raise WorkbenchError("文档正文授权已撤销或过期，请重新确认来源", "DOCUMENT_AUTH_CHANGED")
            else:
                self._read(actor,reference)

    @contextmanager
    def commit_guard(self, actor, task_id):
        # A source mutation cannot race the final checkpoint/result commit.
        with self.knowledge.lock:
            self.assert_valid(actor,task_id)
            yield

    def prepare_context(self, actor, task_id, evidence, budget=6000):
        with self.knowledge.lock:
            self.assert_valid(actor,task_id)
            excerpts=[];used=0;seen=set()
            for item in evidence:
                for field,fact in item["data"].get("local_facts",{}).items():
                    if fact.get("kind")!="document_reference":
                        continue
                    reference=fact["citation"]
                    if reference["chunk_id"] in seen:
                        continue
                    seen.add(reference["chunk_id"])
                    try:
                        approval,resolved=self._consent(actor,reference)
                    except WorkbenchError:
                        # An undisclosed stale candidate is simply unavailable for this request.
                        continue
                    if approval is None:
                        continue
                    size=token_count(resolved["text"])
                    if used+size>budget:
                        continue
                    citation=resolved["citation"]
                    bound={**reference,"document_hash":citation["document_hash"]}
                    with self.research.connection() as c:
                        c.execute("INSERT OR IGNORE INTO research_document_sources VALUES(?,?,?,?,?,?,?)",
                                  (task_id,citation["doc_id"],citation["version"],citation["chunk_id"],approval["id"],
                                   dumps(bound),utc_now()))
                    excerpts.append({"evidence_id":item["evidence_id"],"field":field,"text":resolved["text"],
                                     "citation":citation,"approval_id":approval["id"]})
                    used+=size
            return {"excerpts":excerpts,"estimated_tokens":used,"provider":PROVIDER}

    def on_document_change(self, doc_id, kind):
        self.invalidate(doc_id,force=kind=="delete")

    def invalidate(self, doc_id, *, force=False, approval_id=None):
        with self.knowledge.lock:
            with self.research.connection() as c:
                rows=c.execute("""SELECT DISTINCT t.id,t.owner_id,t.request_json FROM research_tasks t
                    JOIN research_document_sources s ON s.task_id=t.id WHERE s.doc_id=?
                    AND (? IS NULL OR s.approval_id=?)""",(doc_id,approval_id,approval_id)).fetchall()
            for row in rows:
                request=json.loads(row["request_json"])
                actor=Principal(row["owner_id"],request["role"])
                invalid=force or approval_id is not None
                try:
                    actor=self.wb.access.current(request["auth_ref"],row["owner_id"],request["role"])
                except WorkbenchError:
                    invalid=True
                if not invalid:
                    try:
                        with self.knowledge.connection() as c:
                            self.knowledge._document(c,actor,doc_id)
                    except WorkbenchError:
                        invalid=True
                if invalid:
                    self._redact(row["id"],doc_id)

    def _redact(self, task_id, doc_id):
        result={"answer":REDACTED,"model_answer":REDACTED,"kind":"clarification","evidence_ids":[],
                "document_content_removed":True}
        with self.research.connection() as c:
            c.execute("PRAGMA secure_delete=ON")
            c.execute("""UPDATE research_tasks SET result_json=?,checkpoint_json='{}',error_json=?,
                         status='needs_attention',version=version+1,fence=fence+1,worker_id=NULL,lease_until=NULL WHERE id=?""",
                      (dumps(result),dumps({"code":"DOCUMENT_SOURCE_REVOKED","message":REDACTED}),task_id))
            c.execute("UPDATE research_messages SET text=? WHERE task_id=? AND role='assistant'",(REDACTED,task_id))
            c.execute("INSERT INTO research_document_audit(task_id,doc_id,action,at) VALUES(?,?,?,?)",
                      (task_id,doc_id,"remove_source_derived_text",utc_now()))
        # WAL checkpoints remove stale copies of deleted text without retaining source-body backups.
        with self.research.connection() as c:
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def can_send_history(self, actor, task_id):
        try:
            self.assert_valid(actor,task_id)
            return True
        except WorkbenchError:
            return False
