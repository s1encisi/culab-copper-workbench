"""在 2024—2025 的 2,732 个唯一 OOF 事件上验证 V1/LangGraph V2 parity。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

V2_ROOT = Path(__file__).resolve().parents[1]
V2_SRC = V2_ROOT / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from copper_langgraph_v2.bridge import fixed_offline_plan, run_offline_graph  # noqa: E402
from copper_langgraph_v2.graph import LangGraphRunner  # noqa: E402
from copper_langgraph_v2.paths import (  # noqa: E402
    DEVELOPMENT_DATA_DIR,
    P2_RUN_MANIFEST,
    RUNS_DIR,
    SELECTED_MODEL_MANIFEST,
)
from copper_mas.evaluation.workflow_benchmark import (  # noqa: E402
    _make_admission,
    _make_request,
    reconstruct_p2_oof_identity,
)
from copper_mas.models.predictors import load_selected_predictor  # noqa: E402

EXPECTED_EVENTS = 2732


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_development_only(paths: dict[str, Path]) -> None:
    forbidden = ("external_2026", "sealed", "external_2026_once")
    for name, path in paths.items():
        normalized = str(path.resolve()).replace("\\", "/").lower()
        if any(token in normalized for token in forbidden):
            raise ValueError(f"V2 OOF parity 禁止读取外部/封存工件: {name}={path}")


def _block_network():
    original = socket.socket.connect

    def denied(*_args, **_kwargs):
        raise RuntimeError("V2 OOF parity 已强制禁用网络")

    socket.socket.connect = denied
    return original


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 2024—2025 OOF 事件的 V1/LangGraph V2 语义一致性。")
    parser.add_argument(
        "--max-events",
        type=int,
        help="仅验证前 N 条，用于快速验收；省略时运行全部 2,732 条。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS_DIR / "langgraph_oof_parity_v1",
    )
    args = parser.parse_args()
    if args.max_events is not None and not 1 <= args.max_events <= EXPECTED_EVENTS:
        parser.error(f"--max-events 必须在 1 到 {EXPECTED_EVENTS} 之间")
    return args


def main() -> int:
    args = _parse_args()
    input_dir = DEVELOPMENT_DATA_DIR
    paths = {
        "features": input_dir / "core_feature_matrix_v2.csv",
        "admissions": input_dir / "as_of_admission_card_core_v2.csv",
        "training_index": input_dir / "training_evaluation_index_v2.csv",
        "folds": input_dir / "cv_fold_manifest_v2.csv",
        "p2_manifest": P2_RUN_MANIFEST,
        "selected_model": SELECTED_MODEL_MANIFEST,
    }
    _assert_development_only(paths)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OOF parity 输入缺失: {missing}")

    selected_payload = json.loads(paths["selected_model"].read_text(encoding="utf-8"))
    if selected_payload.get("external_2026_read") is not False:
        raise ValueError("冻结模型清单未声明 external_2026_read=false")
    predictor = load_selected_predictor(paths["selected_model"])
    p2_manifest = json.loads(paths["p2_manifest"].read_text(encoding="utf-8"))
    if (p2_manifest.get("safety") or {}).get("external_2026_read") is not False:
        raise ValueError("P2 清单未声明 external_2026_read=false")

    features = pd.read_csv(paths["features"], low_memory=False)
    admissions = pd.read_csv(paths["admissions"], low_memory=False)
    training_index = pd.read_csv(paths["training_index"], low_memory=False)
    folds = pd.read_csv(paths["folds"], low_memory=False)
    for frame, name in ((features, "features"), (admissions, "admissions")):
        if frame["origin_event_id"].duplicated().any():
            raise ValueError(f"{name} origin_event_id 不唯一")
    features["decision_at"] = pd.to_datetime(features["decision_at"], errors="raise", format="mixed")
    admissions["decision_at"] = pd.to_datetime(admissions["decision_at"], errors="raise", format="mixed")
    admissions["feature_cutoff_at"] = pd.to_datetime(admissions["feature_cutoff_at"], errors="raise", format="mixed")

    expected_source_hashes = p2_manifest.get("source_sha256") or {}
    for filename, key in (
        ("core_feature_matrix_v2.csv", "features"),
        ("training_evaluation_index_v2.csv", "training_index"),
        ("cv_fold_manifest_v2.csv", "folds"),
    ):
        actual = _hash(paths[key])
        if expected_source_hashes.get(filename) != actual:
            raise ValueError(f"{filename} 与 P2 冻结输入哈希不一致")

    sample = reconstruct_p2_oof_identity(
        training_index,
        folds,
        p2_manifest,
        expected_events=EXPECTED_EVENTS,
    )
    evaluated_event_count = args.max_events or EXPECTED_EVENTS
    sample = sample.iloc[:evaluated_event_count].copy()
    feature_by_origin = features.set_index("origin_event_id", drop=False)
    admission_by_origin = admissions.set_index("origin_event_id", drop=False)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"OOF parity 输出已存在，拒绝覆盖: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoints.sqlite"
    rows: list[dict[str, Any]] = []
    v1_latencies: list[float] = []
    v2_latencies: list[float] = []
    started = time.perf_counter()
    original_connect = _block_network()
    try:
        with LangGraphRunner(
            checkpoint_path=checkpoint_path,
            predict=predictor,
        ) as runner:
            for _, identity in sample.iterrows():
                pair_id = str(identity["pair_id"])
                origin_event_id = str(identity["origin_event_id"])
                feature_series = feature_by_origin.loc[origin_event_id]
                admission_series = admission_by_origin.loc[origin_event_id]
                if isinstance(feature_series, pd.DataFrame) or isinstance(admission_series, pd.DataFrame):
                    raise ValueError(f"起点 {origin_event_id} 输入不唯一")
                feature_row = feature_series.to_dict()
                admission = _make_admission(admission_series)
                request = _make_request(admission=admission, feature_row=feature_row)
                plan = fixed_offline_plan(
                    run_id=f"V2_OOF_PARITY::{pair_id}",
                    numeric_predictor_ref=predictor.model_id,
                    post_freeze=False,
                )

                v1_start = time.perf_counter()
                expected = run_offline_graph(
                    plan=plan,
                    request=request,
                    feature_row=feature_row,
                    predict=predictor,
                )
                v1_latency = (time.perf_counter() - v1_start) * 1000
                v2_start = time.perf_counter()
                actual = runner.invoke(
                    plan=plan,
                    request=request,
                    feature_row=feature_row,
                )
                v2_latency = (time.perf_counter() - v2_start) * 1000
                v1_latencies.append(v1_latency)
                v2_latencies.append(v2_latency)

                expected_mode = expected.process_mode.model_dump(mode="json")
                expected_prediction = expected.prediction.model_dump(mode="json")
                expected_audit = expected.audit.model_dump(mode="json")
                mode_match = actual.get("process_mode") == expected_mode
                prediction_match = actual.get("prediction") == expected_prediction
                audit_match = actual.get("audit") == expected_audit
                trace_match = actual.get("trace") == ["A1", "A2", "A4", "A5"]
                resource_safe = bool(
                    actual.get("resources")
                    and actual["resources"]["total_calls"] == 0
                    and actual["resources"]["total_input_tokens"] == 0
                    and actual["resources"]["total_output_tokens"] == 0
                    and actual["resources"]["total_estimated_cost_cny"] == 0
                )
                passed = all((mode_match, prediction_match, audit_match, trace_match, resource_safe))
                rows.append(
                    {
                        "pair_id": pair_id,
                        "origin_event_id": origin_event_id,
                        "fold_id": str(identity["fold_id"]),
                        "decision_at": pd.Timestamp(identity["decision_at"]).isoformat(),
                        "mode_match": mode_match,
                        "prediction_match": prediction_match,
                        "audit_match": audit_match,
                        "trace_match": trace_match,
                        "resource_safe": resource_safe,
                        "blocked": bool(actual.get("blocked")),
                        "v1_latency_ms": v1_latency,
                        "v2_latency_ms": v2_latency,
                        "passed": passed,
                    }
                )
                if len(rows) % 250 == 0:
                    print(f"progress={len(rows)}/{evaluated_event_count}", flush=True)
    finally:
        socket.socket.connect = original_connect

    result_frame = pd.DataFrame(rows)
    result_path = output_dir / "oof_event_parity_v1.csv"
    result_frame.to_csv(result_path, index=False, encoding="utf-8-sig")
    mismatch_count = int((~result_frame["passed"]).sum())
    manifest = {
        "run_id": "LANGGRAPH_V2_OOF_PARITY_V1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "development_years": [2024, 2025],
        "sample": "tier1_core_primary_prospective_unique_oof_validation_union",
        "run_scope": "full" if args.max_events is None else "partial_smoke",
        "expected_event_count": EXPECTED_EVENTS,
        "evaluated_event_count": int(len(result_frame)),
        "passed_event_count": int(result_frame["passed"].sum()),
        "mismatch_count": mismatch_count,
        "mode_match_count": int(result_frame["mode_match"].sum()),
        "prediction_match_count": int(result_frame["prediction_match"].sum()),
        "audit_match_count": int(result_frame["audit_match"].sum()),
        "trace_match_count": int(result_frame["trace_match"].sum()),
        "blocked_count": int(result_frame["blocked"].sum()),
        "frozen_model_id": predictor.model_id,
        "v1_latency_ms": {
            "p50": float(np.quantile(v1_latencies, 0.50)),
            "p95": float(np.quantile(v1_latencies, 0.95)),
        },
        "langgraph_v2_latency_ms": {
            "p50": float(np.quantile(v2_latencies, 0.50)),
            "p95": float(np.quantile(v2_latencies, 0.95)),
        },
        "checkpoint_database_bytes": checkpoint_path.stat().st_size,
        "llm_calls_made": 0,
        "network_calls_made": 0,
        "external_2026_read": False,
        "sealed_outcome_read": False,
        "source_sha256": {name: _hash(path) for name, path in paths.items()},
        "output_sha256": {result_path.name: _hash(result_path)},
        "status": "FAILED_PARITY",
    }
    if mismatch_count == 0 and len(result_frame) == evaluated_event_count:
        manifest["status"] = (
            "PASSED_EXACT_SAMPLEWISE_SEMANTIC_PARITY"
            if args.max_events is None
            else "PASSED_PARTIAL_SAMPLEWISE_SEMANTIC_PARITY"
        )
    manifest_path = output_dir / "run_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"].startswith("PASSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
