"""Verify OCR, human corrections and citation lifecycle on the synthetic scan fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from copper_mvp.api import create_app
from copper_mvp.common import write_json
from copper_mvp.data import DataRepository


def corrections(parse_hash):
    return {
        "expected_parse_hash": parse_hash,
        "note": "Compared all text, table cells and formula case with the synthetic source image.",
        "corrections": [
            {
                "block_id": 0,
                "regions": [
                    {"kind": "heading", "text": "合成 OCR 验证页"},
                    {"kind": "paragraph", "text": "仅用于软件测试，数值不代表工厂设置。\n一、浓度记录"},
                    {
                        "kind": "table",
                        "headers": ["字段", "合成值", "单位"],
                        "rows": [["铜浓度", "12.5", "g/L"], ["砷浓度", "24.0", "mg/L"]],
                        "footnotes": ["表注：所有数值均为合成测试样例。"],
                        "bbox": [120, 400, 1350, 765],
                    },
                    {"kind": "heading", "text": "二、表达式及符号"},
                    {
                        "kind": "equation",
                        "text": "c = m / V",
                        "definitions": "c 为质量浓度，m 为溶质质量，V 为溶液体积。",
                        "bbox": [120, 865, 1350, 1130],
                    },
                    {"kind": "paragraph", "text": "识别结果需要核对表头、单位与符号后使用。"},
                ],
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    payload = args.fixture.read_bytes()
    fixture_hash = hashlib.sha256(payload).hexdigest()
    report = {"status": "running", "fixture_sha256": fixture_hash, "scope": "synthetic scan only"}
    paths = [
        *sorted((ROOT / "src/copper_mvp").glob("knowledge_*.py")),
        ROOT / "src/copper_mvp/api_knowledge.py",
        ROOT / "configs/runtime/knowledge_ocr.json",
        Path(__file__),
    ]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report["source_hashes"] = source_hashes
    blocked = []
    original_connect, original_connection = socket.socket.connect, socket.create_connection

    def checked_connect(connection, address):
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return original_connect(connection, address)
        blocked.append(True)
        raise RuntimeError("External network access blocked during OCR verification")

    def checked_connection(address, *rest, **kwargs):
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return original_connection(address, *rest, **kwargs)
        blocked.append(True)
        raise RuntimeError("External network access blocked during OCR verification")

    socket.socket.connect, socket.create_connection = checked_connect, checked_connection
    try:
        with TestClient(create_app(args.output / "runtime", DataRepository()), base_url="http://127.0.0.1") as client:
            key = client.app.state.workbench.access.owner_key_path.read_text(encoding="utf-8").strip()
            assert client.post("/api/auth/session", json={"access_code": key}).status_code == 200
            reader_key = client.post("/api/v2/access-keys", json={"user_id": "reader", "role": "viewer"}).json()[
                "access_code"
            ]
            reader = {"Authorization": "Bearer " + reader_key}
            spec = {
                "doc_id": "scan-fixture",
                "title": "合成扫描测试",
                "author": "test-author",
                "source": "synthetic OCR fixture",
                "authorization": "local software verification",
                "license": "synthetic",
                "version_label": "v1",
                "format": "pdf",
                "effective_at": "2025-01-01T00:00:00Z",
                "readers": ["reader"],
            }
            assert client.post("/api/v2/knowledge/documents", json=spec).status_code == 201
            base = "/api/v2/knowledge/documents/scan-fixture/versions/1"
            upload = client.put(base + "/content", content=payload)
            assert upload.status_code == 200, upload.text
            initial = upload.json()
            assert initial["status"] == "needs_review"
            assert any(v["reason"] == "ocr_required" for v in initial["parsed"]["issues"])
            assert client.post(base + "/index").status_code == 400
            assert (
                client.post(
                    base + "/ocr", json={"expected_parse_hash": initial["parse_hash"]}, headers=reader
                ).status_code
                == 403
            )
            response = client.post(base + "/ocr", json={"expected_parse_hash": initial["parse_hash"]})
            assert response.status_code == 200, response.text
            ocr = response.json()
            ocr_block = ocr["parsed"]["blocks"][0]
            assert "铜浓度" in ocr_block["text"] and "12.5" in ocr_block["text"]
            assert ocr["status"] == "needs_review"
            page = client.get(base + "/pages/1.png")
            assert page.status_code == 200 and page.headers["Cache-Control"] == "no-store"
            (args.output / "review_page.png").write_bytes(page.content)
            fixed = client.post(base + "/corrections", json=corrections(ocr["parse_hash"]))
            assert fixed.status_code == 200, fixed.text
            corrected = fixed.json()
            stale = client.post(
                base + "/review",
                json={"expected_parse_hash": ocr["parse_hash"], "accepted": True, "note": "stale parse"},
            )
            assert stale.status_code == 409
            approved = client.post(
                base + "/review",
                json={
                    "expected_parse_hash": corrected["parse_hash"],
                    "accepted": True,
                    "note": "Verified synthetic image, units and uppercase V.",
                },
            )
            assert approved.status_code == 200, approved.text
            indexed = client.post(base + "/index")
            assert indexed.status_code == 200, indexed.text
            retrieved = client.post(
                "/api/v2/knowledge/search", json={"query": "铜和砷浓度使用什么单位？", "limit": 20}, headers=reader
            )
            assert retrieved.status_code == 200, retrieved.text
            table = next(v for v in retrieved.json()["items"] if v["kind"] == "table")
            assert all(v in table["text"] for v in ("铜浓度", "12.5", "g/L", "砷浓度", "24.0", "mg/L"))
            equation = next(v for v in retrieved.json()["items"] if v["kind"] == "equation")
            assert "c = m / V" in equation["text"] and "V 为溶液体积" in equation["text"]
            assert table["citation"]["parse_hash"] == corrected["parse_hash"]
            assert table["citation"]["location"]["region_bbox"] == [120, 400, 1350, 765]
            history = client.get(base + "/parses").json()["items"]
            assert len(history) == 3
            raw_ocr = client.get(base + "/parses/2", headers=reader).json()
            assert raw_ocr["parsed"]["blocks"][0]["text"] == ocr_block["text"]
            original = client.get(table["citation"]["source_url"], headers=reader)
            assert hashlib.sha256(original.content).hexdigest() == fixture_hash
            assert client.delete("/api/v2/knowledge/documents/scan-fixture").status_code == 200
            assert client.get(base + "/parses/2", headers=reader).status_code == 404
            with client.app.state.workbench.knowledge.connection() as connection:
                assert connection.execute("SELECT count(*) FROM document_parses").fetchone()[0] == 0
            assert not blocked
            assert hashlib.sha256(args.fixture.read_bytes()).hexdigest() == fixture_hash
            report.update(
                status="passed",
                ocr_lines=len(ocr_block["details"]["ocr"]["lines"]),
                ocr_minimum_score=ocr_block["details"]["ocr"]["minimum_score"],
                raw_ocr_text=ocr_block["text"],
                ocr_signature=ocr_block["details"]["ocr"]["engine_signature"],
                table_units_preserved=True,
                formula_case_corrected=True,
                stale_review_rejected=True,
                parse_history_count=len(history),
                deletion_removed_parse_history=True,
                original_unchanged=True,
                outbound_requests=len(blocked),
                source_hashes=source_hashes,
            )
            assert source_hashes == {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            }
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        socket.socket.connect, socket.create_connection = original_connect, original_connection
        write_json(args.output / "verification.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("source_hashes", "raw_ocr_text")}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
