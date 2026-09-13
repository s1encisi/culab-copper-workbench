"""Parse revisions, OCR and explicit human corrections for registered documents."""
from __future__ import annotations

import copy
import json

from copper_mvp.common import WorkbenchError, digest, dumps, utc_now
from copper_mvp.knowledge_ocr import LocalOCR, render_page


class DocumentProcessing:
    def __init__(self, store):
        self.store = store
        self.ocr = LocalOCR()

    def archive(self, c, actor_id, doc_id, version, parsed, operation):
        revision = c.execute("SELECT COALESCE(MAX(revision),0)+1 FROM document_parses WHERE doc_id=? AND version=?",
                             (doc_id, version)).fetchone()[0]
        parsed_hash = digest(parsed)
        c.execute("INSERT INTO document_parses VALUES(?,?,?,?,?,?,?,?)",
                  (doc_id, version, revision, parsed_hash, dumps(parsed), actor_id, operation, utc_now()))
        return parsed_hash

    def initialize_history(self, c):
        rows = c.execute("SELECT doc_id,version,parsed_json,index_id,index_parse_hash FROM document_versions WHERE parsed_json IS NOT NULL").fetchall()
        for row in rows:
            if not c.execute("SELECT 1 FROM document_parses WHERE doc_id=? AND version=?", (row["doc_id"], row["version"])).fetchone():
                self.archive(c, "legacy_migration", row["doc_id"], row["version"], json.loads(row["parsed_json"]),
                             "preserve_existing_parse")
            if row["index_id"] and row["index_parse_hash"] is None:
                c.execute("UPDATE document_versions SET index_parse_hash=? WHERE doc_id=? AND version=?",
                          (digest(json.loads(row["parsed_json"])), row["doc_id"], row["version"]))

    def history(self, actor, doc_id, version):
        with self.store.lock, self.store.connection() as c:
            self.store._document(c, actor, doc_id)
            self.store._version(c, doc_id, version)
            return [dict(row) for row in c.execute(
                "SELECT revision,parse_hash,actor_id,operation,at FROM document_parses WHERE doc_id=? AND version=? ORDER BY revision",
                (doc_id, version))]

    def historical_parse(self, actor, doc_id, version, revision):
        with self.store.lock, self.store.connection() as c:
            self.store._document(c, actor, doc_id)
            row = c.execute("SELECT parse_hash,parsed_json,actor_id,operation,at FROM document_parses WHERE doc_id=? AND version=? AND revision=?",
                            (doc_id, version, revision)).fetchone()
            if row is None:
                raise WorkbenchError("没有该解析修订", "DOCUMENT_NOT_FOUND")
            return {"revision": revision, "parse_hash": row["parse_hash"], "parsed": json.loads(row["parsed_json"]),
                    "actor_id": row["actor_id"], "operation": row["operation"], "at": row["at"]}

    def inventory(self, actor):
        actor.require("manage")
        with self.store.lock, self.store.connection() as c:
            rows = c.execute("""SELECT v.doc_id,v.version,v.metadata_json,v.status,v.index_id,d.revoked,d.acl_json
                                FROM document_versions v JOIN documents d USING(doc_id)
                                WHERE d.project_id=? AND d.deleted_at IS NULL ORDER BY v.doc_id,v.version DESC""",
                             (actor.project_id,)).fetchall()
            return [{**{k: row[k] for k in ("doc_id", "version", "status", "index_id", "revoked")},
                     "metadata": json.loads(row["metadata_json"]), "access": json.loads(row["acl_json"])} for row in rows]

    def _editable(self, c, actor, doc_id, version, expected):
        doc = self.store._document(c, actor, doc_id, manage=True)
        if doc["revoked"]:
            raise WorkbenchError("文档已撤销", "DOCUMENT_REVOKED")
        row = self.store._version(c, doc_id, version)
        if row["parsed_json"] is None:
            raise WorkbenchError("文档尚未解析", "DOCUMENT_STATE")
        parsed = json.loads(row["parsed_json"])
        if digest(parsed) != expected:
            raise WorkbenchError("解析版本已变更，请重新读取后审核", "VERSION_CONFLICT")
        return row, parsed

    def run_ocr(self, actor, doc_id, version, expected_parse_hash, pages=None, dpi=180):
        with self.store.lock, self.store.connection() as c:
            row, parsed = self._editable(c, actor, doc_id, version, expected_parse_hash)
            if json.loads(row["metadata_json"])["format"] != "pdf":
                raise WorkbenchError("此入口处理 PDF 扫描页", "DOCUMENT_FORMAT")
            selected = pages or sorted({v["location"]["page"] for v in parsed["issues"] if v["reason"] == "ocr_required"})
            if not selected:
                raise WorkbenchError("没有待识别扫描页，请指定需要重新识别的页码", "DOCUMENT_OCR_PAGES")
            if len(selected) > 20 or len(set(selected)) != len(selected):
                raise WorkbenchError("每批选择不重复的 1 至 20 页", "DOCUMENT_OCR_PAGES")
            content = bytes(row["content"])
        # Long inference does not hold the store lock. Recheck ACL and revision before publishing.
        replacements = {page: self.ocr.recognize(content, page, dpi) for page in selected}
        revised = copy.deepcopy(parsed)
        revised["blocks"] = [v for v in revised["blocks"] if v["location"].get("page") not in replacements]
        revised["blocks"] += list(replacements.values())
        revised["blocks"].sort(key=lambda v: v["location"].get("page") or 0)
        for index, item in enumerate(revised["blocks"]):
            item["block_id"] = index
        revised["issues"] = [v for v in revised["issues"] if v["location"].get("page") not in replacements]
        for page, item in replacements.items():
            revised["issues"].append({"location": item["location"],
                                      "reason": "ocr_review" if item["text"].strip() else "ocr_no_text"})
        revised["parser_version"] = "g6m.structured-ocr.v2"
        with self.store.lock, self.store.connection() as c:
            self._editable(c, actor, doc_id, version, expected_parse_hash)
            self.archive(c, actor.user_id, doc_id, version, revised, "local_ocr")
            c.execute("UPDATE document_versions SET parsed_json=?,status='needs_review',review_json=NULL WHERE doc_id=? AND version=?",
                      (dumps(revised), doc_id, version))
            self.store._audit(c, actor, doc_id, "local_ocr")
            return self.store._describe(self.store._version(c, doc_id, version))

    def correct(self, actor, doc_id, version, request):
        with self.store.lock, self.store.connection() as c:
            row, parsed = self._editable(c, actor, doc_id, version, request.expected_parse_hash)
            corrections = {item.block_id: item for item in request.corrections}
            if len(corrections) != len(request.corrections) or set(corrections) - {v["block_id"] for v in parsed["blocks"]}:
                raise WorkbenchError("修订需引用不同的现有结构块", "DOCUMENT_CORRECTION")
            revised = copy.deepcopy(parsed)
            output, resolved_locations = [], []
            for original in revised["blocks"]:
                correction = corrections.get(original["block_id"])
                if correction is None:
                    output.append(original)
                    continue
                resolved_locations.append(original["location"])
                for region in correction.regions:
                    location = dict(original["location"])
                    if region.bbox is not None:
                        width, height = location.get("rendered_size", location.get("pdf_size_points", [0, 0]))
                        x0, y0, x1, y1 = region.bbox
                        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                            raise WorkbenchError("修订区域超出原页面范围", "DOCUMENT_CORRECTION")
                        location["region_bbox"] = region.bbox
                    details = {"correction_source": "human_review", "actor_id": actor.user_id,
                               "original_block_id": original["block_id"], "original_parse_hash": request.expected_parse_hash,
                               "reason": request.note, "executable": False}
                    text = region.text
                    if region.kind == "table":
                        if not region.rows or not region.headers:
                            raise WorkbenchError("表格修订需包含表头、单位及完整数据行", "DOCUMENT_CORRECTION")
                        if any(len(row) != len(region.headers) for row in region.rows):
                            raise WorkbenchError("表格数据行与表头列数不一致", "DOCUMENT_CORRECTION")
                        details.update(headers=region.headers, rows=region.rows, footnotes=region.footnotes,
                                       merged_cells=region.merged_cells)
                        text = " | ".join(region.headers) + "\n" + "\n".join(" | ".join(r) for r in region.rows)
                        if region.footnotes:
                            text += "\n" + "\n".join(region.footnotes)
                    if region.kind == "equation":
                        details["symbol_definitions"] = region.definitions
                        if not region.definitions:
                            raise WorkbenchError("公式修订需同时保留符号定义", "DOCUMENT_CORRECTION")
                        text += "\n" + region.definitions
                    if not text.strip():
                        raise WorkbenchError("修订块不能为空", "DOCUMENT_CORRECTION")
                    output.append({"kind": region.kind, "text": text, "location": location,
                                   "heading": region.heading or original["heading"], "details": details,
                                   "authority": "evidence_only"})
            for i, item in enumerate(output):
                item["block_id"] = i
            revised["blocks"] = output
            revised["issues"] = [v for v in revised["issues"] if v["location"] not in resolved_locations]
            revised["issues"].append({"location": {}, "reason": "human_corrections_review"})
            revised["parser_version"] = "g6m.structured-ocr.v2"
            self.archive(c, actor.user_id, doc_id, version, revised, "human_correction")
            c.execute("UPDATE document_versions SET parsed_json=?,status='needs_review',review_json=NULL WHERE doc_id=? AND version=?",
                      (dumps(revised), doc_id, version))
            self.store._audit(c, actor, doc_id, "human_correction")
            return self.store._describe(self.store._version(c, doc_id, version))

    def page_image(self, actor, doc_id, version, page, as_of=None):
        content, format = self.store.source(actor, doc_id, version, as_of)
        if format != "pdf":
            raise WorkbenchError("仅 PDF 提供页图", "DOCUMENT_FORMAT")
        return render_page(content, page)[0]
