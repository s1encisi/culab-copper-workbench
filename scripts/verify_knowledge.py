"""Offline integration evaluation on explicitly synthetic knowledge fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi.testclient import TestClient

from copper_mvp.api import create_app
from copper_mvp.common import write_json
from copper_mvp.data import DataRepository

FIXTURES = {
    "units": ("合成变量表", "铜质量浓度使用 g/L，砷质量浓度使用 mg/L。更换单位时需要显式换算。"),
    "labels": (
        "合成时间合同",
        "实验室化验结果只有发布完成后才能作为标签使用。迟到标签不能回填到早于发布时间的历史决策中。",
    ),
    "validation": (
        "合成模型协议",
        "按时间顺序设置训练集和验证集。预处理和特征选择只在训练样本内拟合，验证结果不参与训练。",
    ),
    "control": (
        "合成控制规程",
        "优化器输出候选方案。候选经审批与联锁检查后才可进入模拟命令执行。文档本身不能授予设备写入权限。",
    ),
    "rollback": (
        "合成发布流程",
        "模型发布时记录工件版本、审批和影子验证。回退使用此前登记的模型工件，恢复后重新核对状态。",
    ),
    "missing": (
        "合成观测合同",
        "传感器缺失时保留缺失标记和观测年龄。向后匹配历史信号，不允许使用决策时刻以后的未来值。",
    ),
}
QUESTIONS = [
    ("铜含量和砷含量分别用什么量纲表示？", "units"),
    ("不同浓度单位可以直接比较吗？", "units"),
    ("迟到化验能否算作过去已知结果？", "labels"),
    ("实验室测量何时可以用于评价？", "labels"),
    ("标准化应该使用哪一部分样本？", "validation"),
    ("验证集能不能参与特征筛选？", "validation"),
    ("优化建议什么时候能变成执行命令？", "control"),
    ("检索到的操作规程可以给予控制权限吗？", "control"),
    ("切换模型出现问题后怎样恢复先前版本？", "rollback"),
    ("发布模型需要留下什么记录？", "rollback"),
    ("传感器没有读数时应记录什么？", "missing"),
    ("匹配输入信号能使用未来的数据吗？", "missing"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Choose a fresh output directory to retain earlier evidence")
    args.output.mkdir(parents=True)
    blocked, loopback_connections = [], []
    original_connect, original_connection = socket.socket.connect, socket.create_connection

    def checked_connect(connection, address):
        # Windows asyncio implements its self-pipe using a loopback socket pair.
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            loopback_connections.append(True)
            return original_connect(connection, address)
        blocked.append(True)
        raise RuntimeError("External network access is forbidden during local knowledge verification")

    def checked_connection(address, *args, **kwargs):
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return original_connection(address, *args, **kwargs)
        blocked.append(True)
        raise RuntimeError("External network access is forbidden during local knowledge verification")

    socket.socket.connect = checked_connect
    socket.create_connection = checked_connection
    paths = sorted((ROOT / "src/copper_mvp").glob("knowledge_*.py"))
    paths += [
        ROOT / "src/copper_mvp/api_knowledge.py",
        ROOT / "configs/runtime/knowledge_embedding.json",
        Path(__file__),
    ]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = {
        "status": "running",
        "scope": "synthetic offline integration; not a reviewed industrial RAG benchmark",
        "source_hashes": source_hashes,
    }
    try:
        data = DataRepository()
        with TestClient(create_app(args.output / "runtime", data), base_url="http://127.0.0.1") as client:
            key = client.app.state.workbench.access.owner_key_path.read_text(encoding="utf-8").strip()
            assert client.post("/api/auth/session", json={"access_code": key}).status_code == 200
            reader_key = client.post("/api/v2/access-keys", json={"user_id": "reader", "role": "viewer"}).json()[
                "access_code"
            ]
            headers = {"Authorization": "Bearer " + reader_key}
            for doc_id, (title, text) in FIXTURES.items():
                metadata = {
                    "doc_id": doc_id,
                    "title": title,
                    "author": "synthetic-fixture",
                    "source": "verify_knowledge.py synthetic fixture",
                    "authorization": "local test only",
                    "license": "synthetic fixture",
                    "classification": "internal",
                    "version_label": "v1",
                    "format": "md",
                    "effective_at": "2025-01-01T00:00:00Z",
                    "readers": ["reader"],
                }
                response = client.post("/api/v2/knowledge/documents", json=metadata)
                assert response.status_code == 201, response.text
                base = f"/api/v2/knowledge/documents/{doc_id}/versions/1"
                raw = (title + "\n\n" + text).encode()
                assert client.put(base + "/content", content=raw).status_code == 200
                response = client.post(base + "/index")
                assert response.status_code == 200, response.text
            records, timings = [], []
            citation_checks = 0
            for question, expected in QUESTIONS:
                start = perf_counter()
                response = client.post(
                    "/api/v2/knowledge/search", json={"query": question, "limit": 20}, headers=headers
                )
                timings.append((perf_counter() - start) * 1000)
                assert response.status_code == 200, response.text
                assert response.headers["Cache-Control"] == "no-store"
                result = response.json()
                rankings = {}
                for channel, field in (("hybrid", "score"), ("bm25", "bm25"), ("dense", "cosine")):
                    ordered = sorted(result["items"], key=lambda v: -v[field])
                    docs = list(dict.fromkeys(v["citation"]["doc_id"] for v in ordered))
                    rankings[channel] = docs
                records.append({"question": question, "expected": expected, "ranking": rankings})
                for item in result["items"][:5]:
                    citation = item["citation"]
                    located = client.get("/api/v2/knowledge/citations/" + citation["chunk_id"], headers=headers)
                    assert located.status_code == 200 and located.json()["text"] == item["text"]
                    source = client.get(citation["source_url"], headers=headers)
                    assert source.status_code == 200
                    assert hashlib.sha256(source.content).hexdigest() == citation["document_hash"]
                    citation_checks += 1
            denied = client.post("/api/v2/knowledge/documents", json=metadata, headers=headers)
            assert denied.status_code == 403
            before = client.post("/api/v2/knowledge/search", json={"query": "铜浓度单位"}, headers=headers).json()
            withdrawn = [v for v in before["items"] if v["citation"]["doc_id"] == "units"]
            assert withdrawn
            revoked = client.put(
                "/api/v2/knowledge/documents/units/access", json={"revoked": True, "readers": ["reader"]}
            )
            assert revoked.status_code == 200
            after = client.post(
                "/api/v2/knowledge/search", json={"query": "铜浓度单位", "limit": 20}, headers=headers
            ).json()
            assert all(v["citation"]["doc_id"] != "units" for v in after["items"])
            assert (
                client.get(
                    "/api/v2/knowledge/citations/" + withdrawn[0]["citation"]["chunk_id"], headers=headers
                ).status_code
                == 404
            )
            metrics = {}
            for channel in ("hybrid", "bm25", "dense"):
                metrics[channel] = {
                    f"recall_at_{k}": float(np.mean([v["expected"] in v["ranking"][channel][:k] for v in records]))
                    for k in (1, 5)
                }
            report.update(
                status="passed",
                queries=len(records),
                documents=len(FIXTURES),
                retrieval=metrics,
                citation_checks=citation_checks,
                citation_correctness=1.0,
                unauthorized_document_leaks=0,
                revoked_citation_readable=False,
                latency_ms={"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95))},
                outbound_requests=len(blocked),
                local_loopback_connections=len(loopback_connections),
                records=records,
                embedding=client.app.state.workbench.knowledge.embedder.manifest,
                remaining=["reviewed domain evaluation", "scan OCR", "session memory and assistant integration"],
            )
            assert not blocked
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
        json.dumps(
            {k: v for k, v in report.items() if k not in ("records", "embedding", "source_hashes")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
