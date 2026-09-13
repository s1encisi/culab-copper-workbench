"""Local document parsing with original source locations and review boundaries."""
from __future__ import annotations

import io
import re
import zipfile

from copper_mvp.common import WorkbenchError

PARSER_VERSION = "g6m.structured.v1"
TOKEN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
MAX_BYTES = 16 * 1024 * 1024


def token_count(text):
    return len(TOKEN.findall(text))


def block(kind, text, location, heading="", **details):
    return {"kind": kind, "text": text, "location": location, "heading": heading,
            "details": details, "authority": "evidence_only"}


def markdown_blocks(payload):
    text = payload.decode("utf-8-sig")
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result, heading, i = [], [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        start = i
        h = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", lines[i])
        if h:
            level = len(h[1])
            heading = heading[:level - 1] + [h[2]]
            kind = "heading"
            i += 1
        elif i + 1 < len(lines) and "|" in lines[i] and re.fullmatch(r"[\s|:\-]+", lines[i + 1]):
            kind = "table"
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                i += 1
        elif lines[i].strip().startswith("$$"):
            kind = "equation"
            i += 1
            if lines[start].strip() == "$$" or not lines[start].strip().endswith("$$"):
                while i < len(lines):
                    closed = "$$" in lines[i]
                    i += 1
                    if closed:
                        break
            # Keep adjacent symbol definitions with the expression.
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith("#"):
                i += 1
        elif lines[i].lstrip().startswith(("~~~", chr(96) * 3)):
            kind = "code"
            fence = lines[i].strip()[:3]
            i += 1
            while i < len(lines):
                closed = lines[i].lstrip().startswith(fence)
                i += 1
                if closed:
                    break
        else:
            kind = "image" if re.search(r"!\[[^\]]*\]\([^)]+\)", lines[i]) else "paragraph"
            i += 1
            while i < len(lines) and lines[i].strip():
                if re.match(r"^(#{1,6})\s|^\$\$|^!\[", lines[i]):
                    break
                if i + 1 < len(lines) and "|" in lines[i] and re.fullmatch(r"[\s|:\-]+", lines[i + 1]):
                    break
                i += 1
        raw = "".join(lines[start:i]).strip()
        details = {}
        if kind == "table":
            rows = [line.strip().strip("|").split("|") for line in lines[start:i]]
            details = {"headers": [v.strip() for v in rows[0]],
                       "rows": [[v.strip() for v in row] for row in rows[2:]],
                       "row_numbers": list(range(start + 3, i + 1)), "unit_context": raw}
        elif kind == "image":
            details = {"references": re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", raw),
                       "description_source": "document", "external_assets_fetched": False}
        elif kind == "equation":
            details = {"expression_and_definitions": raw, "executable": False}
        result.append(block(kind, raw, {"page": None, "line_start": start + 1,
                            "line_end": i, "char_start": offsets[start], "char_end": offsets[i]},
                            " / ".join(heading), **details))
    return result, []


def docx_blocks(payload):
    from docx import Document
    from lxml import etree
    from docx.table import Table
    from docx.oxml.ns import qn
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 128 * 1024 * 1024:
            raise WorkbenchError("解压后的文档过大", "DOCUMENT_SIZE")
    document = Document(io.BytesIO(payload))
    result, issues, heading = [], [], ""
    for index, item in enumerate(document.iter_inner_content(), 1):
        location = {"page": None, "body_block": index, "xml_part": "word/document.xml"}
        if isinstance(item, Table):
            rows, grid = [], []
            for row_index, row in enumerate(item.rows, 1):
                rows.append([cell.text for cell in row.cells])
                cells = []
                for tc in row._tr.tc_lst:
                    span = tc.tcPr.gridSpan if tc.tcPr is not None else None
                    merge = tc.tcPr.vMerge if tc.tcPr is not None else None
                    cells.append({"grid_span": int(span.val) if span is not None else 1,
                                  "vertical_merge": str(merge.val) if merge is not None else None})
                grid.append({"row": row_index, "cells": cells})
            text = "\n".join(" | ".join(row) for row in rows)
            result.append(block("table", text, location, heading, rows=rows,
                                headers=rows[0] if rows else [], merged_cells=grid))
            continue
        style = item.style.name if item.style is not None else ""
        if style.startswith("Heading"):
            heading = item.text
        if item.text.strip():
            result.append(block("heading" if style.startswith("Heading") else "paragraph",
                                item.text, location, heading))
        equations = item._p.xpath(".//m:oMath")
        for equation in equations:
            text = "".join(equation.itertext())
            result.append(block("equation", text + "\n" + item.text, location, heading, xml=etree.tostring(equation, encoding="unicode"),
                                executable=False, context=item.text))
            issues.append({"location": location, "reason": "equation_review"})
        for drawing in item._p.xpath(".//w:drawing"):
            refs = list(drawing.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"))
            images = []
            for ref in refs:
                rid = ref.get(qn("r:embed"))
                if rid and rid in document.part.rels:
                    images.append(str(document.part.rels[rid].target_ref))
            result.append(block("image", item.text or "文档内嵌图片", location, heading,
                                original_parts=images, description_source="document"))
            issues.append({"location": location, "reason": "image_description_review"})
    for rel in document.part.rels.values():
        if not rel.is_external and rel.reltype.endswith(("/footnotes", "/endnotes")):
            root = etree.fromstring(rel.target_part.blob, etree.XMLParser(resolve_entities=False, no_network=True))
            for note in root:
                text = " ".join(note.itertext()).strip()
                if text:
                    result.append(block("footnote", text, {"xml_part": str(rel.target_ref),
                                        "note_id": note.get(qn("w:id")), "page": None}, heading))
    return result, issues


def pdf_blocks(payload):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(payload))
    if reader.is_encrypted:
        raise WorkbenchError("请先提供经授权解密的 PDF", "DOCUMENT_ENCRYPTED")
    result, issues = [], []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text(extraction_mode="layout") or ""
        location = {"page": number, "bbox": [float(v) for v in page.mediabox]}
        if text.strip():
            result.append(block("page", text.strip(), location, extracted_by="pypdf-layout"))
            issues.append({"location": location, "reason": "pdf_layout_table_equation_review"})
        else:
            result.append(block("image", "", location, original_page=number, ocr_confidence=None))
            issues.append({"location": location, "reason": "ocr_required" if len(page.images) else "blank_or_vector_page_review"})
    return result, issues


def parse_document(payload, format):
    if not payload or len(payload) > MAX_BYTES:
        raise WorkbenchError("文档大小需在 1 字节至 16 MB 之间", "DOCUMENT_SIZE")
    parser = {"md": markdown_blocks, "docx": docx_blocks, "pdf": pdf_blocks}.get(format)
    if parser is None:
        raise WorkbenchError("支持 MD、DOCX 和 PDF 文档", "DOCUMENT_FORMAT")
    try:
        blocks, issues = parser(payload)
    except (WorkbenchError, ImportError):
        raise
    except Exception as exc:
        raise WorkbenchError("文档解析失败，请检查文件格式", "DOCUMENT_PARSE") from exc
    if not blocks:
        raise WorkbenchError("文档没有可解析的内容", "DOCUMENT_EMPTY")
    for i, item in enumerate(blocks):
        item["block_id"] = i
    return {"parser_version": PARSER_VERSION, "blocks": blocks, "issues": issues}


def make_chunks(blocks, target=480, overlap=80):
    """Preserve tables/equations as units; long prose keeps explicit character spans."""
    result = []
    for item in blocks:
        text = item["text"]
        if not text.strip():
            continue
        structural = item["kind"] in {"table", "equation", "image", "code"}
        spans = list(TOKEN.finditer(text))
        starts = [0] if structural else range(0, max(len(spans), 1), target - overlap)
        for start in starts:
            stop = len(spans) if structural else min(start + target, len(spans))
            left = spans[start].start() if spans else 0
            right = spans[stop - 1].end() if spans else len(text)
            fragment = text[left:right]
            result.append({"text": fragment, "heading": item["heading"], "kind": item["kind"],
                           "block_id": item["block_id"], "location": item["location"],
                           "span": [left, right], "tokens": token_count(fragment)})
            if stop == len(spans):
                break
    return result
