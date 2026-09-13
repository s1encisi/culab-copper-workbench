"""Project-scoped documents and atomic local hybrid indexes."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from urllib.parse import quote

import numpy as np

from copper_mvp.common import WorkbenchError, digest, dumps, utc_now
from copper_mvp.knowledge_contracts import DocumentSpec, DocumentAccess
from copper_mvp.knowledge_embedding import LocalEmbedding, load_dependencies
from copper_mvp.knowledge_parsing import parse_document, make_chunks
from copper_mvp.knowledge_processing import DocumentProcessing

INDEX_VERSION = "g6m.bm25-rrf.v2"


def terms(text):
    pieces = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower())
    result = []
    for piece in pieces:
        if re.fullmatch(r"[\u3400-\u9fff]+", piece):
            result.extend(piece)
            result.extend(piece[i:i + 2] for i in range(len(piece) - 1))
        else:
            result.append(piece)
    return result


def timestamp(value=None):
    moment = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise WorkbenchError("知识查询时间必须包含时区", "KNOWLEDGE_TIME")
    return moment.astimezone(timezone.utc).isoformat()


class KnowledgeStore:
    def __init__(self, root, embedder=None):
        self.root = Path(root) / "knowledge"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "knowledge.sqlite"
        self.lock = threading.RLock()
        self.change_listeners = []
        self.embedder = embedder or LocalEmbedding()
        self.processing = DocumentProcessing(self)
        with self.connection() as c:
            c.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                    acl_json TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT);
                CREATE TABLE IF NOT EXISTS document_versions (
                    doc_id TEXT NOT NULL, version INTEGER NOT NULL, metadata_json TEXT NOT NULL,
                    content BLOB, content_hash TEXT, parsed_json TEXT, status TEXT NOT NULL,
                    review_json TEXT, index_id TEXT, embedding_signature TEXT,
                    PRIMARY KEY(doc_id,version),
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id));
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, version INTEGER NOT NULL,
                    index_id TEXT NOT NULL, record_json TEXT NOT NULL, vector BLOB NOT NULL,
                    FOREIGN KEY(doc_id,version) REFERENCES document_versions(doc_id,version) ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS knowledge_by_version ON knowledge_chunks(doc_id,version,index_id);
                CREATE TABLE IF NOT EXISTS document_parses (
                    doc_id TEXT NOT NULL, version INTEGER NOT NULL, revision INTEGER NOT NULL,
                    parse_hash TEXT NOT NULL, parsed_json TEXT NOT NULL, actor_id TEXT NOT NULL,
                    operation TEXT NOT NULL, at TEXT NOT NULL,
                    PRIMARY KEY(doc_id,version,revision),
                    FOREIGN KEY(doc_id,version) REFERENCES document_versions(doc_id,version) ON DELETE CASCADE);
                CREATE TABLE IF NOT EXISTS knowledge_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL, action TEXT NOT NULL, at TEXT NOT NULL);
            """)

            columns = {row["name"] for row in c.execute("PRAGMA table_info(document_versions)")}
            if "index_parse_hash" not in columns:
                c.execute("ALTER TABLE document_versions ADD COLUMN index_parse_hash TEXT")
            if "index_version" not in columns:
                c.execute("ALTER TABLE document_versions ADD COLUMN index_version TEXT")
                c.execute("UPDATE document_versions SET index_version='g6m.bm25-rrf.v1' WHERE index_id IS NOT NULL")
            self.processing.initialize_history(c)

    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.db_path, timeout=20)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA secure_delete=ON")
        try:
            with c:
                yield c
        finally:
            c.close()

    @staticmethod
    def _audit(c, actor, doc_id, action):
        c.execute("INSERT INTO knowledge_audit(doc_id,actor_id,action,at) VALUES(?,?,?,?)",
                  (doc_id, actor.user_id, action, utc_now()))

    def _document(self, c, actor, doc_id, *, manage=False):
        actor.require("manage" if manage else "read")
        row = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None or row["deleted_at"] or row["project_id"] != actor.project_id:
            raise WorkbenchError("没有可访问的文档", "DOCUMENT_NOT_FOUND")
        acl = json.loads(row["acl_json"])
        permitted = actor.role == "owner" or actor.user_id in acl["readers"] or actor.role in acl["read_roles"]
        if not manage and (row["revoked"] or not permitted):
            raise WorkbenchError("没有可访问的文档", "DOCUMENT_NOT_FOUND")
        return row

    @staticmethod
    def _version(c, doc_id, version):
        row = c.execute("SELECT * FROM document_versions WHERE doc_id=? AND version=?", (doc_id, version)).fetchone()
        if row is None:
            raise WorkbenchError("没有该文档版本", "DOCUMENT_NOT_FOUND")
        return row

    def register(self, actor, spec: DocumentSpec):
        actor.require("manage")
        if actor.project_id != spec.project_id:
            raise WorkbenchError("不能向其他项目登记文档", "FORBIDDEN")
        metadata = spec.model_dump(mode="json", exclude={"readers", "read_roles"})
        metadata["effective_at"] = timestamp(metadata["effective_at"])
        if metadata["expires_at"]:
            metadata["expires_at"] = timestamp(metadata["expires_at"])
        with self.lock, self.connection() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id=?", (spec.doc_id,)).fetchone()
            if row is None:
                c.execute("INSERT INTO documents(doc_id,project_id,acl_json) VALUES(?,?,?)",
                          (spec.doc_id, spec.project_id, dumps({"readers": spec.readers, "read_roles": spec.read_roles})))
            else:
                self._document(c, actor, spec.doc_id, manage=True)
                if row["revoked"]:
                    raise WorkbenchError("文档已撤销，请先检查访问设置", "DOCUMENT_REVOKED")
                acl = json.loads(row["acl_json"])
                if acl != {"readers": spec.readers, "read_roles": spec.read_roles}:
                    raise WorkbenchError("版本更新不改变权限，请先通过访问设置更新", "DOCUMENT_ACL")
            versions = c.execute("SELECT version,metadata_json FROM document_versions WHERE doc_id=?", (spec.doc_id,)).fetchall()
            if any(json.loads(v["metadata_json"])["version_label"] == spec.version_label for v in versions):
                raise WorkbenchError("文档版本标签已存在", "VERSION_CONFLICT")
            version = max((v["version"] for v in versions), default=0) + 1
            c.execute("INSERT INTO document_versions(doc_id,version,metadata_json,status) VALUES(?,?,?,'registered')",
                      (spec.doc_id, version, dumps(metadata)))
            self._audit(c, actor, spec.doc_id, "register")
        return {"doc_id": spec.doc_id, "version": version, "status": "registered"}

    def inspect(self, actor, doc_id, version):
        with self.lock, self.connection() as c:
            self._document(c, actor, doc_id)
            row = self._version(c, doc_id, version)
            return self._describe(row)

    @staticmethod
    def _describe(row):
        return {"doc_id": row["doc_id"], "version": row["version"], "metadata": json.loads(row["metadata_json"]),
                "content_hash": row["content_hash"], "status": row["status"],
                "parsed": json.loads(row["parsed_json"]) if row["parsed_json"] else None,
                "review": json.loads(row["review_json"]) if row["review_json"] else None,
                "index_id": row["index_id"], "index_parse_hash": row["index_parse_hash"],
                "parse_hash": digest(json.loads(row["parsed_json"])) if row["parsed_json"] else None}

    def authorize_upload(self, actor, doc_id, version):
        with self.lock, self.connection() as c:
            row = self._document(c, actor, doc_id, manage=True)
            if row["revoked"]:
                raise WorkbenchError("文档已撤销", "DOCUMENT_REVOKED")
            self._version(c, doc_id, version)

    def upload(self, actor, doc_id, version, payload):
        # ACL is checked before dependencies are loaded or the parser receives bytes.
        with self.lock, self.connection() as c:
            self.authorize_upload(actor, doc_id, version)
            row = self._version(c, doc_id, version)
            content_hash = hashlib.sha256(payload).hexdigest()
            if row["status"] != "registered":
                if row["content_hash"] == content_hash:
                    return self._describe(row)
                raise WorkbenchError("已登记的文件不可覆盖，请创建新版本", "VERSION_CONFLICT")
            load_dependencies()
            parsed = parse_document(payload, json.loads(row["metadata_json"])["format"])
            status = "needs_review" if parsed["issues"] else "parsed"
            c.execute("UPDATE document_versions SET content=?,content_hash=?,parsed_json=?,status=? WHERE doc_id=? AND version=?",
                      (payload, content_hash, dumps(parsed), status, doc_id, version))
            self.processing.archive(c, actor.user_id, doc_id, version, parsed, "parse")
            self._audit(c, actor, doc_id, "parse")
            return self._describe(self._version(c, doc_id, version))

    def review(self, actor, doc_id, version, accepted, note, expected_parse_hash=None):
        with self.lock, self.connection() as c:
            self._document(c, actor, doc_id, manage=True)
            row = self._version(c, doc_id, version)
            if row["status"] not in {"parsed", "needs_review", "rejected"}:
                raise WorkbenchError("该版本当前不能审核", "DOCUMENT_STATE")
            parsed = json.loads(row["parsed_json"])
            parse_hash = digest(parsed)
            if expected_parse_hash is not None and expected_parse_hash != parse_hash:
                raise WorkbenchError("解析版本已变更，请重新读取后审核", "VERSION_CONFLICT")
            if accepted and any(issue["reason"] in {"ocr_required", "ocr_no_text"} for issue in parsed["issues"]):
                raise WorkbenchError("扫描页尚无 OCR 文字，不能直接标记为已解析", "DOCUMENT_OCR_REQUIRED")
            review = {"accepted": accepted, "note": note, "actor_id": actor.user_id, "at": utc_now(),
                      "content_hash": row["content_hash"], "parse_hash": parse_hash}
            c.execute("UPDATE document_versions SET status=?,review_json=? WHERE doc_id=? AND version=?",
                      ("parsed" if accepted else "rejected", dumps(review), doc_id, version))
            self._audit(c, actor, doc_id, "review")
            return self._describe(self._version(c, doc_id, version))

    def build_index(self, actor, doc_id, version):
        with self.lock, self.connection() as c:
            doc = self._document(c, actor, doc_id, manage=True)
            if doc["revoked"]:
                raise WorkbenchError("已撤销文档不能建立索引", "DOCUMENT_REVOKED")
            row = self._version(c, doc_id, version)
            if row["status"] not in {"parsed", "indexed"}:
                raise WorkbenchError("文档需要完成解析或审核", "DOCUMENT_STATE")
            signature = self.embedder.signature
            parsed = json.loads(row["parsed_json"])
            parse_hash = digest(parsed)
            if row["status"] == "indexed" and row["embedding_signature"] == signature and row["index_parse_hash"] == parse_hash and row["index_version"] == INDEX_VERSION:
                return {"doc_id": doc_id, "version": version, "index_id": row["index_id"], "reused": True}
            parsed = json.loads(row["parsed_json"])
            chunks = make_chunks(parsed["blocks"])
            if not chunks:
                raise WorkbenchError("没有可索引的文字", "DOCUMENT_EMPTY")
            vectors = self.embedder.encode([v["text"] for v in chunks])
            index_id = digest({"content_hash": row["content_hash"], "parser": parsed["parser_version"],
                               "embedding": signature, "index_version": INDEX_VERSION, "parse_hash": parse_hash})
            c.execute("DELETE FROM knowledge_chunks WHERE doc_id=? AND version=?", (doc_id, version))
            for item, vector in zip(chunks, vectors):
                content_hash = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                chunk_id = digest({"doc_id": doc_id, "version": version, "block": item["block_id"],
                                   "span": item["span"], "row_range": item.get("row_range"), "hash": content_hash, "parse_hash": parse_hash})[:32]
                item.update(chunk_id=chunk_id, hash=content_hash, document_hash=row["content_hash"], parse_hash=parse_hash)
                c.execute("INSERT INTO knowledge_chunks VALUES(?,?,?,?,?,?)",
                          (chunk_id, doc_id, version, index_id, dumps(item), np.asarray(vector, dtype="<f4").tobytes()))
            c.execute("UPDATE document_versions SET index_id=?,embedding_signature=?,index_parse_hash=?,index_version=?,status='indexed' WHERE doc_id=? AND version=?",
                      (index_id, signature, parse_hash, INDEX_VERSION, doc_id, version))
            self._audit(c, actor, doc_id, "index")
            return {"doc_id": doc_id, "version": version, "index_id": index_id,
                    "chunks": len(chunks), "embedding_signature": signature, "reused": False}

    def notify_change(self, doc_id, kind):
        for listener in tuple(self.change_listeners):
            listener(doc_id, kind)

    def change_access(self, actor, doc_id, access: DocumentAccess):
        with self.lock, self.connection() as c:
            self._document(c, actor, doc_id, manage=True)
            c.execute("UPDATE documents SET acl_json=?,revoked=? WHERE doc_id=?",
                      (dumps(access.model_dump(exclude={"revoked"})), int(access.revoked), doc_id))
            self._audit(c, actor, doc_id, "revoke" if access.revoked else "access_change")
        self.notify_change(doc_id, "access")
        return {"doc_id": doc_id, "revoked": access.revoked}

    def _visible(self, c, actor, as_of):
        actor.require("read")
        selected = {}
        rows = c.execute("""SELECT v.doc_id,v.version,v.metadata_json,v.index_id,v.embedding_signature,v.index_parse_hash,v.index_version,d.acl_json
                            FROM document_versions v JOIN documents d USING(doc_id)
                            WHERE d.project_id=? AND d.revoked=0 AND d.deleted_at IS NULL AND v.index_id IS NOT NULL""",
                         (actor.project_id,)).fetchall()
        for row in rows:
            acl = json.loads(row["acl_json"])
            if actor.role != "owner" and actor.user_id not in acl["readers"] and actor.role not in acl["read_roles"]:
                continue
            meta = json.loads(row["metadata_json"])
            if meta["effective_at"] > as_of:
                continue
            previous = selected.get(row["doc_id"])
            if previous is None or (meta["effective_at"], row["version"]) > (
                    json.loads(previous["metadata_json"])["effective_at"], previous["version"]):
                selected[row["doc_id"]] = row
        return {key: row for key, row in selected.items()
                if not json.loads(row["metadata_json"])["expires_at"]
                or json.loads(row["metadata_json"])["expires_at"] > as_of}

    def list(self, actor, as_of=None):
        with self.lock, self.connection() as c:
            rows = self._visible(c, actor, timestamp(as_of))
            return [{"doc_id": row["doc_id"], "version": row["version"],
                     "metadata": json.loads(row["metadata_json"]), "index_id": row["index_id"]}
                    for row in rows.values()]

    @staticmethod
    def citation(item, row, as_of):
        return {"doc_id": row["doc_id"], "version": row["version"], "chunk_id": item["chunk_id"],
                "page": item["location"].get("page"), "heading": item["heading"],
                "location": item["location"], "span": item["span"], "hash": item["hash"],
                "document_hash": item["document_hash"], "index_id": row["index_id"], "as_of": as_of,
                "parse_hash": item.get("parse_hash", row["index_parse_hash"]), "index_version": row["index_version"],
                "source_url": f"/api/v2/knowledge/documents/{row['doc_id']}/versions/{row['version']}/source?as_of={quote(as_of, safe='')}"
                              + (f"#page={item['location']['page']}" if item["location"].get("page") else "")}

    def cache_signature(self, actor, as_of=None):
        with self.lock, self.connection() as c:
            visible = self._visible(c, actor, timestamp(as_of))
            revision = c.execute("SELECT COALESCE(MAX(id),0) FROM knowledge_audit").fetchone()[0]
            return digest({"revision": revision,
                           "visible": sorted((row["doc_id"], row["version"], row["index_id"]) for row in visible.values())})

    def search(self, actor, query, as_of=None, limit=5, context_tokens=6000):
        bound = timestamp(as_of)
        with self.lock, self.connection() as c:
            visible = self._visible(c, actor, bound)
            chunks, vectors, versions = [], [], []
            # Fetch text/vector only after project, ACL and effective-time selection.
            for row in visible.values():
                if row["embedding_signature"] != self.embedder.signature:
                    raise WorkbenchError("嵌入版本已变更，需要重建索引", "KNOWLEDGE_INDEX_VERSION")
                for chunk in c.execute("SELECT * FROM knowledge_chunks WHERE doc_id=? AND version=? AND index_id=?",
                                       (row["doc_id"], row["version"], row["index_id"])):
                    chunks.append(json.loads(chunk["record_json"]))
                    vectors.append(np.frombuffer(chunk["vector"], dtype="<f4"))
                    versions.append(row)
            if not chunks:
                return {"items": [], "as_of": bound, "context_tokens": 0, "index_version": INDEX_VERSION}
            counters = [Counter(terms(item["heading"] + " " + item["text"])) for item in chunks]
            lengths = np.asarray([sum(counter.values()) for counter in counters], dtype=float)
            avg = max(float(lengths.mean()), 1)
            scores = np.zeros(len(chunks))
            for term in set(terms(query)):
                df = sum(term in counter for counter in counters)
                idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
                tf = np.asarray([counter[term] for counter in counters], dtype=float)
                scores += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * lengths / avg))
            semantic = np.asarray(vectors) @ self.embedder.encode([query], query=True)[0]
            lexical = [int(i) for i in np.argsort(-scores, kind="stable")[:20] if scores[i] > 0]
            dense = [int(i) for i in np.argsort(-semantic, kind="stable")[:20]]
            fused = Counter()
            for channel in (lexical, dense):
                for rank, index in enumerate(channel, 1):
                    fused[index] += 1 / (60 + rank)
            items, used = [], 0
            for index, score in sorted(fused.items(), key=lambda pair: (-pair[1], pair[0])):
                item, row = chunks[index], versions[index]
                if used + item["tokens"] > context_tokens:
                    continue
                items.append({"text": item["text"], "kind": item["kind"], "score": score,
                              "bm25": float(scores[index]), "cosine": float(semantic[index]),
                              "citation": self.citation(item, row, bound), "authority": "evidence_only"})
                used += item["tokens"]
                if len(items) >= limit:
                    break
            return {"items": items, "as_of": bound, "context_tokens": used, "index_version": INDEX_VERSION,
                    "reranking": "reciprocal_rank_fusion_k60", "embedding_signature": self.embedder.signature,
                    "local_only": True}

    def resolve(self, actor, chunk_id, as_of=None):
        bound = timestamp(as_of)
        with self.lock, self.connection() as c:
            visible = self._visible(c, actor, bound)
            for row in visible.values():
                item = c.execute("SELECT record_json FROM knowledge_chunks WHERE chunk_id=? AND doc_id=? AND version=? AND index_id=?",
                                 (chunk_id, row["doc_id"], row["version"], row["index_id"])).fetchone()
                if item:
                    chunk = json.loads(item["record_json"])
                    if hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest() != chunk["hash"]:
                        raise WorkbenchError("引用内容哈希不一致", "KNOWLEDGE_CITATION_HASH")
                    return {"text": chunk["text"], "citation": self.citation(chunk, row, bound), "authority": "evidence_only"}
            raise WorkbenchError("引用已失效、撤销或不再可访问", "CITATION_NOT_FOUND")

    def source(self, actor, doc_id, version, as_of=None):
        with self.lock, self.connection() as c:
            self._document(c, actor, doc_id)
            visible = self._visible(c, actor, timestamp(as_of))
            row = visible.get(doc_id)
            if actor.role != "owner" and (row is None or row["version"] != version):
                raise WorkbenchError("该文档版本在查询时刻不可访问", "DOCUMENT_NOT_FOUND")
            row = self._version(c, doc_id, version)
            if row["content"] is None:
                raise WorkbenchError("该版本还没有原文", "DOCUMENT_NOT_FOUND")
            raw = bytes(row["content"])
            if hashlib.sha256(raw).hexdigest() != row["content_hash"]:
                raise WorkbenchError("原文哈希不一致", "KNOWLEDGE_SOURCE_HASH")
            return raw, json.loads(row["metadata_json"])["format"]

    def delete(self, actor, doc_id):
        with self.lock:
            with self.connection() as c:
                self._document(c, actor, doc_id, manage=True)
                c.execute("DELETE FROM document_versions WHERE doc_id=?", (doc_id,))
                c.execute("UPDATE documents SET deleted_at=?,revoked=1,acl_json='{}' WHERE doc_id=?", (utc_now(), doc_id))
                self._audit(c, actor, doc_id, "delete_content_and_vectors")
            with self.connection() as c:
                c.execute("VACUUM")
        self.notify_change(doc_id, "delete")
        return {"doc_id": doc_id, "deleted": True, "retained": "audit_tombstone_only"}
