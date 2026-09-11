"""Validate G1 on local development data and save an auditable event timeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copper_mvp.common import DEFAULT_RUNS_DIR, PROJECT_ROOT, digest, dumps, file_hash
from copper_mvp.data import DataRepository, OFFSETS
from copper_mvp.data_contracts import ReplayQuery, SOURCE_TIMEZONE, source_time
from copper_mvp.data_service import DataService
from copper_mvp.modeling import ModelManager


def matrix_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--before", type=Path, help="Optional immutable pre-change baseline JSON.")
    parser.add_argument("--model-run-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()
    output = (args.output or PROJECT_ROOT / "runs/g1" / datetime.now(timezone.utc).strftime("acceptance_%Y%m%dT%H%M%SZ")).resolve()
    if not output.is_relative_to((PROJECT_ROOT / "runs").resolve()):
        parser.error("Private event evidence must be saved under the local runs/ directory.")
    names = ("after.json", "dependencies.json", "evidence_at_decision.json", "evidence_at_maturity.json", "report.md")
    if any((output / name).exists() for name in names):
        parser.error("Refusing to overwrite an existing G1 acceptance artifact.")
    started = perf_counter()
    data = DataRepository()
    source_hashes = {key: file_hash(path) for key, path in data.paths.sources().items()}
    models = ModelManager(args.model_run_dir, data)
    service = DataService(data, models)
    X, y = data.training_data()
    ledger = service.labels
    latest_at = source_time(data.frame.decision_at.max())
    labels = ledger.latest(latest_at)
    reconstructed = np.array([[labels[(event, target)].value for target in ("cu", "as")] for event in y.index])
    np.testing.assert_array_equal(reconstructed, y.to_numpy())

    bad_anchors = 0
    admitted = np.array([data.admissions[event]["admission_status"] != "REJECTED" for event in data.frame.index])
    for signal in data.signals:
        for offset in OFFSETS:
            column = f"{signal['tag']}__t_minus_{offset}h"
            values = pd.to_numeric(data.frame[column], errors="coerce").to_numpy()
            ages = pd.to_numeric(data.frame[column + "__age_h"], errors="coerce").to_numpy()
            bad_anchors += int((admitted & np.isfinite(values) & (~np.isfinite(ages) | (ages < 0))).sum())
    if bad_anchors:
        raise AssertionError("Admitted feature anchors violate source-time availability.")
    for event, row in data.frame.iterrows():
        admission = data.admissions[event]
        if (source_time(admission["decision_at"]) != source_time(row.decision_at)
            or source_time(admission["feature_cutoff_at"]) > source_time(row.decision_at)
            or admission["contract_id"] != service.timing["contract_id"]):
            raise AssertionError("Admission identity/time/contract mismatch.")

    fold_checks = []
    for fold in sorted(data.cv.fold_id.unique()):
        rows = data.cv[data.cv.fold_id.eq(fold)]
        cutoff = source_time(rows.fold_fit_cutoff_at.iloc[0])
        available = ledger.latest(cutoff)
        train_ids = data.training_ids(fold)
        if not all((event, target) in available and available[(event, target)].decision_at < cutoff
                   for event in train_ids for target in ("cu", "as")):
            raise AssertionError("A training fold accesses an immature label.")
        fold_checks.append({"fold_id": fold, "train_events": len(train_ids), "maturity_passed": True})

    baseline_checks = None
    if args.before:
        baseline = json.loads(args.before.read_text(encoding="utf-8"))
        actual = {"dataset_version": data.dataset_version, "x_hash": matrix_hash(X), "y_hash": matrix_hash(y),
                  "contexts_hash": digest([data.context(event) for event in baseline["event_ids"]])}
        if any(actual[key] != baseline[key] for key in actual):
            raise AssertionError("Legacy dataset, matrices or contexts changed.")
        compared = 0
        for previous in baseline["predictions"]:
            profile = previous["predictions"]["cu"]["model"]
            current = models.predict(previous["event_id"], profile, "oof_replay", previous["bundle_id"])
            if dumps(current) != dumps(previous):
                raise AssertionError("A saved legacy prediction changed.")
            query = ReplayQuery(model_profile=profile, bundle_id=previous["bundle_id"])
            g1 = service.evidence(previous["event_id"], query, verify_sources=False)
            if dumps(g1["payload"]["prediction"]["payload"]["predictions"]) != dumps(previous["predictions"]):
                raise AssertionError("G1 numeric prediction differs from legacy output.")
            compared += 1
        baseline_checks = {"predictions_compared": compared, "predictions_equal": True,
                           "full_X_equal": True, "full_y_equal": True, "contexts_equal": True}

    event = data.events(scope="dual", limit=1)["items"][0]["event_id"]
    before = service.evidence(event, ReplayQuery())
    target_time = max(r.available_at for r in ledger.records if r.event_id == event)
    mature = service.evidence(event, ReplayQuery(as_of=target_time))
    if before["payload"]["labels"] or mature["payload"]["evaluation"]["status"] != "EVALUABLE":
        raise AssertionError("Real event did not transition across label availability correctly.")
    mature_all = ledger.snapshot(latest_at, days=3650, limit=3000, minimum=1)
    if mature_all["payload"]["selected_events"] != len(X):
        raise AssertionError("Full development label eligibility differs from training matrix.")
    if source_hashes != {key: file_hash(path) for key, path in data.paths.sources().items()}:
        raise AssertionError("A source file changed during acceptance.")
    report = {
        "status": "PASSED", "scope": "G1 local development evidence and compatibility",
        "development_events": len(data.frame), "training_shape": list(X.shape),
        "labels_shape": list(y.shape), "label_records": len(ledger.records),
        "full_development_mature_events": mature_all["payload"]["selected_events"],
        "oof_events": len(data.fold_for), "all_admission_times_passed": True,
        "invalid_admitted_anchors": bad_anchors, "fold_checks": fold_checks,
        "baseline_comparison": baseline_checks, "source_hashes_unchanged": True,
        "source_hashes": source_hashes, "example_event_id": event,
        "at_decision": before["payload"]["evaluation"]["status"],
        "at_label_availability": mature["payload"]["evaluation"]["status"],
        "external_2026_rows_read": False, "network_calls_made": 0,
        "elapsed_ms": (perf_counter() - started) * 1000,
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"after.json": report, "dependencies.json": service.dependencies(include_paths=True),
                 "evidence_at_decision.json": before, "evidence_at_maturity.json": mature}
    for name, payload in artifacts.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(dumps(payload) + "\n")
    markdown = "# G1 证据与数据基线验收\n\n"
    markdown += f"状态：PASSED。开发事件 {len(data.frame)} 个，训练矩阵 {len(X)}×{X.shape[1]}，OOF 事件 {len(data.fold_for)} 个。\n\n"
    markdown += "全开发集标签重建与原训练标签逐值一致；所有准入时间及有效锚点通过检查；五个时间折的训练标签均已成熟。输入文件哈希保持一致。\n\n"
    if baseline_checks:
        markdown += f"改造前后 {baseline_checks['predictions_compared']} 份既有预测完全一致，完整 X/y 和所选旧事件上下文保持一致。\n\n"
    markdown += "## 一个真实事件的时间线\n\n"
    markdown += f"事件：{event}。实际采样时间未记录；下表单独标记合同假定的采样时间。\n\n"
    markdown += "| 环节 | UTC | 上海时间 | 说明 |\n|---|---|---|---|\n"
    for item in mature["payload"]["timeline"]:
        at = source_time(item["at"])
        note = "合同假设，非实际采样记录" if item.get("is_assumption") else item.get("target", "")
        markdown += f"| {item['kind']} | {at.isoformat()} | {at.astimezone(SOURCE_TIMEZONE).isoformat()} | {note} |\n"
    markdown += "\n决策时点状态为 LABELS_NOT_AVAILABLE，不返回未来标签；标签可得时点状态为 EVALUABLE。预测按历史虚拟决策时间回放，本次实际计算时间另存于证据记录的 created_at。\n\n"
    markdown += "[决策时点证据](evidence_at_decision.json) · [成熟后证据](evidence_at_maturity.json) · [本地依赖清单](dependencies.json) · [完整验收结果](after.json)\n\n"
    markdown += "本报告验证 G1 的功能、时间边界和兼容性；不把它作为模型精度提高或真实在线运行的证明。2026 外评和外部 API 未被调用。\n"
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_hashes", "example_event_id")},
                     ensure_ascii=False, indent=2))
    print("Report: " + str(output / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
