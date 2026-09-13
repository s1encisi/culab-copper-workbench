"""Run fixed-method, fixed-seed studies through the existing comparison service."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import time

import pandas as pd

from copper_mvp.classical_study import evaluate_classical_study
from copper_mvp.common import PROJECT_ROOT, WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.ensemble_experiment import process_peak_mb
from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_evaluation import read_training
from copper_mvp.model_registry import METHOD_IDS, ComparisonRequest, method_spec
from copper_mvp.model_training import source_signature

SOURCE_FILES = tuple("src/copper_mvp/" + name for name in (
    "model_inventory_study.py", "common.py", "model_training.py", "model_evaluation.py",
    "model_registry.py", "model_adapters.py", "model_comparisons.py",
    "specialized_registry.py", "specialized_models.py", "specialized_evaluation.py",
    "cubist_model.py", "classical_models.py", "classical_registry.py",
    "classical_evaluation.py", "classical_study.py", "ensemble_evaluation.py",
)) + ("requirements-specialized-models.txt", "requirements-cubist.txt")


def run_inventory_study(data, run_root, reference_id, request_key, methods, seeds,
                        progress=lambda value: None, max_wall_seconds=1800):
    run_root = Path(run_root)
    methods, seeds = tuple(methods), tuple(seeds)
    service = ComparisonService(run_root / "model_comparisons", data)
    reference = service.get(reference_id)
    if reference["status"] != "completed" or reference["result"]["status"] != "completed":
        raise WorkbenchError("参考比较未完整完成", "INVENTORY_REFERENCE")
    reference_root = service.directory(reference_id)
    old_manifest, old_protocol = read_training(reference_root)
    source = source_signature(data)
    if source != old_manifest["source"]:
        raise WorkbenchError("参考比较的数据版本不同", "SOURCE_CHANGED")
    baseline_methods = [m for m in old_protocol["request"]["methods"]
                        if m != "Persistence" and m not in methods]
    reference_seed = old_protocol["request"]["seed"]
    expected = {"Persistence": [seeds[0]], **{m: list(seeds) for m in methods},
                **{m: [reference_seed] for m in baseline_methods}}
    root = run_root / "model_inventory_studies" / digest(request_key)[:32]
    root.mkdir(parents=True, exist_ok=True)
    source_files = SOURCE_FILES
    if "BART" in methods:
        from copper_mvp.bart_registry import BART_SOURCE_FILES
        source_files += BART_SOURCE_FILES
    if "SymbolicRegression" in methods:
        from copper_mvp.symbolic_registry import SYMBOLIC_SOURCE_FILES
        source_files += SYMBOLIC_SOURCE_FILES
    if any(m in ("TabNet", "FTTransformer", "NODE") for m in methods):
        from copper_mvp.tabular_registry import TABULAR_SOURCE_FILES
        source_files += TABULAR_SOURCE_FILES
    if any(method_spec(m, seeds[0]).get("input_kind") == "anchored_process_sequence" for m in methods):
        from copper_mvp.temporal_registry import TEMPORAL_FILES
        source_files += TEMPORAL_FILES
    if "TabPFN" in methods:
        from copper_mvp.tabpfn_registry import TABPFN_FILES
        source_files += TABPFN_FILES
    code_hashes = {p: file_hash(PROJECT_ROOT / p) for p in source_files}
    protocol = {
        "schema_version": "model-inventory-study.g6g.v1", "request_key": request_key,
        "methods": [method_spec(m, seeds[0]) for m in methods], "seeds": list(seeds),
        "expected_method_seeds": expected, "source": source, "code_hashes": code_hashes,
        "reference_comparison_id": reference_id, "reference_seed": reference_seed,
        "reference_training_hash": file_hash(reference_root / "training_manifest.json"),
        "reference_oof_hash": old_manifest["oof_sha256"],
        "evaluation_as_of": old_protocol["evaluation_as_of"],
        "design": "Original five forward folds; fixed parameters; independent two-target scoring.",
        "replication": f"{len(seeds)} declared seeds per new method; original baseline seed retained.",
        "max_wall_seconds_per_comparison": max_wall_seconds, "native_threads": 1,
        "external_2026_read": False, "automatic_promotion": False,
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        stored = json.loads(protocol_path.read_text(encoding="utf-8"))
        if {k: v for k, v in stored.items() if k != "created_at"} != protocol:
            raise WorkbenchError("研究协议已改变，请使用新请求键", "REQUEST_CONFLICT")
        protocol = stored
    else:
        protocol["created_at"] = utc_now()
        write_json(protocol_path, protocol)
        for path in source_files:
            destination = root / "executed_source" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((PROJECT_ROOT / path).read_bytes())
    state = {"id": root.name, "status": "running", "owner_pid": os.getpid(),
             "created_at": protocol["created_at"], "comparisons": []}
    write_json(root / "state.json", state)
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        for seed in seeds:
            request = ComparisonRequest(request_key=f"{request_key}-seed-{seed}",
                methods=("Persistence", *methods), seed=seed, max_wall_seconds=max_wall_seconds)
            with ThreadPoolExecutor(max_workers=1) as executor:
                current = service.submit(request, executor)
                identifier = current["run_id"]
                state.update(current_seed=seed, current_comparison_id=identifier)
                write_json(root / "state.json", state)
                previous = None
                while True:
                    current = service.get(identifier, result=False)
                    if current.get("progress") != previous:
                        previous = current.get("progress")
                        state["progress"] = previous
                        write_json(root / "state.json", state)
                        progress({"seed": seed, "comparison_id": identifier, "progress": previous})
                    if current["status"] not in ("queued", "running"):
                        break
                    time.sleep(.25)
            current = service.get(identifier)
            state["comparisons"].append({"seed": seed, "comparison_id": identifier,
                                         "status": current["status"]})
            write_json(root / "state.json", state)
            if current["status"] != "completed" or current["result"]["status"] != "completed":
                raise WorkbenchError("方法比较未完整完成，工件与失败保留", "INVENTORY_COMPARISON")
        if source_signature(data) != source or any(file_hash(PROJECT_ROOT / p) != h
                                                  for p, h in code_hashes.items()):
            raise WorkbenchError("研究期间数据或实现改变", "INVENTORY_SOURCE_CHANGED")
        frames, resources, uncertainty, bindings = [], [], [], []
        for item in state["comparisons"]:
            folder = service.directory(item["comparison_id"])
            manifest, comparison_protocol = read_training(folder)
            result = service.get(item["comparison_id"])["result"]
            if comparison_protocol["evaluation_as_of"] != protocol["evaluation_as_of"]:
                raise WorkbenchError("比较的评价截止时间不同", "INVENTORY_EVALUATION_TIME")
            frame = pd.read_csv(folder / "oof_predictions.csv")
            frame["seed"] = item["seed"]
            frames.append(frame)
            for destination, key in ((resources, "resources"), (uncertainty, "uncertainty_metrics")):
                destination.extend({**r, **item} for r in result.get(key, []))
            bindings.append({**item, "training_sha256": file_hash(folder / "training_manifest.json"),
                "oof_sha256": manifest["oof_sha256"], "protocol_sha256": manifest["protocol_sha256"],
                "evaluation_sha256": file_hash(folder / "evaluation.json")})
        old = pd.read_csv(reference_root / "oof_predictions.csv")
        old = old[old.method_id.isin(baseline_methods)].copy()
        old["seed"] = reference_seed
        old["source_scope"] = "original_G2a_reference"
        frames.append(old)
        table = pd.concat(frames, ignore_index=True)
        table = table[~(table.method_id.eq("Persistence") & table.seed.ne(seeds[0]))]
        table.to_csv(root / "predictions.csv", index=False)
        result = evaluate_classical_study(data, table, protocol)
        result.update(schema_version="model-inventory-evaluation.g6g.v1",
            new_method_count=len(methods), registered_total=len(METHOD_IDS),
            compared_methods=len(expected), resources=resources, uncertainty_metrics=uncertainty,
            reference_resources=reference["result"]["resources"], source_comparisons=bindings,
            protocol_sha256=file_hash(protocol_path), predictions_sha256=file_hash(root / "predictions.csv"),
            elapsed_ms=(time.perf_counter()-started)*1000, cpu_seconds=time.process_time()-cpu_started,
            peak_process_rss_mb=process_peak_mb(), runtime={"python": platform.python_version(), "native_threads": 1},
            automatic_promotion=False, created_at=utc_now())
        write_json(root / "evaluation.json", safe(result))
        pd.DataFrame(result["metrics"]).to_csv(root / "metrics.csv", index=False)
        lines = ["# 专用预测方法的固定时间折比较", "",
                 f"新增方法：{', '.join(methods)}。种子：{', '.join(map(str, seeds))}。",
                 f"共同评价 {result['common_events']} 个事件。参数在评价前固定，参考方法沿用原比较的真实预测和原种子。", "",
                 "| 方法 | 目标 | MAE 均值 | 种子间 SD | 相对 Persistence 的 MAE 差 |",
                 "| --- | --- | ---: | ---: | ---: |"]
        for row in result["metrics"]:
            sd = "—" if row["seed_mae_sd"] is None else f"{row['seed_mae_sd']:.6g}"
            lines.append(f"| {row['method_id']} | {row['target']} | {row['mae']:.6g} | {sd} | {row['paired_mae_difference']:.6g} |")
        lines += ["", "概率指标、逐种子结果、时间块配对区间、分工况误差与资源记录见完整评价。不确定性按各方法的登记合同报告，尚未进行独立时序校准。",
                  "", "[完整评价](evaluation.json) · [固定协议](protocol.json) · [逐事件预测](predictions.csv)", ""]
        (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
        state.update(status=result["status"], finished_at=utc_now(),
                     evaluation_sha256=file_hash(root / "evaluation.json"))
        write_json(root / "state.json", state)
        return root, result
    except Exception as exc:
        state.update(status="failed", finished_at=utc_now(),
                     error={"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)[:300]})
        write_json(root / "state.json", state)
        raise
