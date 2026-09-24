"""Classical method inventory study: all methods once, stochastic methods on five seeds."""

from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from copper_mvp.classical_registry import CLASSICAL_METHODS, STOCHASTIC_METHODS
from copper_mvp.common import WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.ensemble_evaluation import hierarchical_interval
from copper_mvp.ensemble_experiment import process_peak_mb
from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_evaluation import numerical_metrics, read_training
from copper_mvp.model_registry import ComparisonRequest, method_spec
from copper_mvp.model_training import source_signature

CLASSICAL_FILES = (
    "classical_registry.py",
    "classical_models.py",
    "classical_evaluation.py",
    "classical_study.py",
    "model_registry.py",
    "model_adapters.py",
    "model_training.py",
    "model_evaluation.py",
    "model_comparisons.py",
)


def run_classical_study(data, run_root, source_comparison_id, request_key, seeds, progress=lambda value: None):
    run_root = Path(run_root)
    service = ComparisonService(run_root / "model_comparisons", data)
    source_state = service.get(source_comparison_id)
    if source_state["status"] != "completed":
        raise WorkbenchError("原六方法比较尚未完成", "CLASSICAL_SOURCE")
    source_root = service.directory(source_comparison_id)
    old_manifest, old_protocol = read_training(source_root)
    if source_signature(data) != old_manifest["source"]:
        raise WorkbenchError("原比较的数据来源不同", "SOURCE_CHANGED")
    identifier = digest(request_key)[:32]
    root = run_root / "classical_studies" / identifier
    root.mkdir(parents=True, exist_ok=True)
    methods_by_seed = {
        str(seed): ["Persistence", *(CLASSICAL_METHODS if i == 0 else STOCHASTIC_METHODS)]
        for i, seed in enumerate(seeds)
    }
    expected = {m: [seeds[0]] for m in old_protocol["request"]["methods"]}
    expected.update({m: list(seeds) if m in STOCHASTIC_METHODS else [seeds[0]] for m in CLASSICAL_METHODS})
    protocol = {
        "schema_version": "classical-study.g6a.v1",
        "request_key": request_key,
        "seeds": list(seeds),
        "source": old_manifest["source"],
        "reference_comparison_id": source_comparison_id,
        "reference_training_hash": file_hash(source_root / "training_manifest.json"),
        "reference_oof_hash": old_manifest["oof_sha256"],
        "methods_by_seed": methods_by_seed,
        "expected_method_seeds": expected,
        "methods": [method_spec(m, seeds[0]) for m in CLASSICAL_METHODS],
        "evaluation_as_of": old_protocol["evaluation_as_of"],
        "code_hashes": {n: file_hash(Path(__file__).with_name(n)) for n in CLASSICAL_FILES},
        "design": "Original five forward outer folds; fixed parameters declared before evaluation; no 2026 data.",
        "replication": "Five seeds for stochastic methods; one fit protocol for deterministic methods.",
        "preprocessing": "New methods use training-only numerical-noise masks; legacy methods remain unchanged.",
        "evaluation_mode": "historical_replay",
        "automatic_promotion": False,
        "new_method_count": len(CLASSICAL_METHODS),
    }
    if (root / "protocol.json").exists():
        stored = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        if digest({k: v for k, v in stored.items() if k != "created_at"}) != digest(protocol):
            raise WorkbenchError("研究协议或实现已改变，请使用新请求键", "REQUEST_CONFLICT")
        protocol = stored
    else:
        protocol["created_at"] = utc_now()
        write_json(root / "protocol.json", protocol)
    state = {
        "id": identifier,
        "status": "running",
        "created_at": protocol["created_at"],
        "owner_pid": os.getpid(),
        "comparisons": [],
    }
    write_json(root / "state.json", state)
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for seed in seeds:
            request = ComparisonRequest(
                request_key=request_key + "-seed-" + str(seed),
                methods=tuple(methods_by_seed[str(seed)]),
                seed=seed,
                max_wall_seconds=1800,
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                current = service.submit(request, executor)
                comparison_id = current["run_id"]
                state["current_seed"] = seed
                state["current_comparison_id"] = comparison_id
                write_json(root / "state.json", state)
                previous = None
                while True:
                    current = service.get(comparison_id, result=False)
                    if current.get("progress") != previous:
                        previous = current.get("progress")
                        state["progress"] = previous
                        write_json(root / "state.json", state)
                        progress({"seed": seed, "comparison_id": comparison_id, "progress": previous})
                    if current["status"] not in ("queued", "running"):
                        break
                    time.sleep(0.2)
            current = service.get(comparison_id)
            state["comparisons"].append({"seed": seed, "comparison_id": comparison_id, "status": current["status"]})
            write_json(root / "state.json", state)
            if current["status"] != "completed":
                raise WorkbenchError("方法比较未完成，已保留失败与工件", "CLASSICAL_COMPARISON_FAILED")
        if source_signature(data) != protocol["source"]:
            raise WorkbenchError("研究期间数据来源改变", "SOURCE_CHANGED")
        frames = []
        resources = []
        uncertainty = []
        sources = []
        for item in state["comparisons"]:
            folder = service.directory(item["comparison_id"])
            record = service.get(item["comparison_id"])
            manifest, comparison_protocol = read_training(folder)
            frame = pd.read_csv(folder / "oof_predictions.csv")
            frame["seed"] = item["seed"]
            frames.append(frame)
            resources.extend(
                {**row, "seed": item["seed"], "comparison_id": item["comparison_id"]}
                for row in record["result"]["resources"]
            )
            uncertainty.extend(
                {**row, "seed": item["seed"], "comparison_id": item["comparison_id"]}
                for row in record["result"].get("uncertainty_metrics", [])
            )
            sources.append(
                {
                    "comparison_id": item["comparison_id"],
                    "seed": item["seed"],
                    "training_manifest_sha256": file_hash(folder / "training_manifest.json"),
                    "protocol_sha256": file_hash(folder / "protocol.json"),
                    "oof_sha256": manifest["oof_sha256"],
                    "evaluation_sha256": file_hash(folder / "evaluation.json"),
                }
            )
        old = pd.read_csv(source_root / "oof_predictions.csv")
        old = old[old.method_id.ne("Persistence")].copy()
        old["seed"] = seeds[0]
        old["source_scope"] = "G2a_reference_predictions"
        frames.append(old)
        table = pd.concat(frames, ignore_index=True)
        # Persistence is a deterministic reference; its repeated copies agree and do not become extra method replicates.
        duplicate_reference = table.method_id.eq("Persistence") & table.seed.ne(seeds[0])
        table = table[~duplicate_reference]
        table.to_csv(root / "predictions.csv", index=False)
        result = evaluate_classical_study(data, table, protocol)
        result.update(
            created_at=utc_now(),
            resources=resources,
            uncertainty_metrics=uncertainty,
            source_comparisons=sources,
            protocol_sha256=file_hash(root / "protocol.json"),
            predictions_sha256=file_hash(root / "predictions.csv"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            cpu_seconds=time.process_time() - cpu_started,
            peak_process_rss_mb=process_peak_mb(),
            runtime={"python": platform.python_version(), "native_threads": 1},
            automatic_promotion=False,
        )
        write_json(root / "evaluation.json", safe(result))
        pd.DataFrame(result["metrics"]).to_csv(root / "metrics.csv", index=False)
        report = "# G6a 经典模型目录比较\n\n"
        report += (
            "新增 "
            f"{len(CLASSICAL_METHODS)}"
            " 种方法，共同评价 "
            f"{result['common_events']}"
            " 个事件；随机方法使用 "
            f"{len(seeds)}"
            " 个预先固定种子，确定性方法只运行一次。\n"
            "\n"
        )
        report += "| 方法 | 目标 | MAE 均值 | 种子间 SD | 相对 Persistence 的 MAE 差 |\n|---|---|---:|---:|---:|\n"
        for row in result["metrics"]:
            sd = "—" if row["seed_mae_sd"] is None else f"{row['seed_mae_sd']:.6g}"
            report += (
                "| "
                f"{row['method_id']}"
                " | "
                f"{row['target']}"
                " | "
                f"{row['mae']:.6g}"
                " | "
                f"{sd}"
                " | "
                f"{row['paired_mae_difference']:.6g}"
                " |\n"
            )
        report += (
            "\n"
            "参数为本轮预先固定配置，预处理和 PCA 只在各训练段拟合。新方法的数"
            "值噪声掩码不回写旧六方法或原数据。概率输出仅报告边际假设和实际覆盖"
            "，不声称已校准或联合覆盖。\n"
            "\n"
        )
        report += (
            "[完整评价](evaluation.json) · [固定协议](protocol.json) · [逐事件"
            "预测](predictions.csv) · [指标 CSV](metrics.csv)\n"
        )
        (root / "report.md").write_bytes(report.encode("utf-8"))
        state.update(
            status=result["status"], finished_at=utc_now(), evaluation_sha256=file_hash(root / "evaluation.json")
        )
        write_json(root / "state.json", state)
        return root, result
    except Exception as exc:
        state.update(
            status="failed",
            finished_at=utc_now(),
            error={"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)[:300]},
        )
        write_json(root / "state.json", state)
        raise


def evaluate_classical_study(data, table, protocol):
    methods = protocol["expected_method_seeds"]
    expected = {(e, m, s) for e in data.fold_for for m, seeds in methods.items() for s in seeds}
    if (
        table.duplicated(["event_id", "method_id", "seed"]).any()
        or set(zip(table.event_id, table.method_id, table.seed, strict=True)) != expected
    ):
        raise WorkbenchError("方法、种子或事件集合不完整", "CLASSICAL_COVERAGE")
    labels = DataService(data).labels.latest(protocol["evaluation_as_of"])
    common = {
        e
        for e in data.fold_for
        if all(
            (e, t) in labels and labels[(e, t)].quality_eligible and labels[(e, t)].value is not None
            for t in ("cu", "as")
        )
    }
    groups = {}
    coverage = {}
    for (method, seed), part in table.groupby(["method_id", "seed"]):
        rows = part.set_index("event_id")
        good = rows.status.eq("completed") & np.isfinite(rows[["cu", "as"]].to_numpy(float)).all(axis=1)
        valid = set(rows.index[good])
        common &= valid
        groups[(method, seed)] = rows
        coverage[f"{method}:{seed}"] = {
            "expected": len(data.fold_for),
            "valid": len(valid),
            "missing_or_failed": len(data.fold_for) - len(valid),
        }
    events = sorted(common, key=lambda e: (data.row(e).decision_at, e))
    if not events:
        raise WorkbenchError("没有共同可评分事件", "CLASSICAL_NO_EVALUATION")
    baseline = groups[("Persistence", protocol["seeds"][0])].loc[events]
    dates = [source_time(data.row(e).decision_at).isoformat() for e in events]
    rows = []
    seed_rows = []
    intervals = {}
    group_metrics = {}
    for method, seeds in methods.items():
        for target, unit in (("cu", "g/L"), ("as", "mg/L")):
            truth = np.array([labels[(e, target)].value for e in events])
            predicted = np.stack([groups[(method, seed)].loc[events, target].to_numpy(float) for seed in seeds])
            errors = np.abs(predicted - truth)
            base = np.abs(baseline[target].to_numpy() - truth)
            means = errors.mean(axis=1)
            for seed, values in zip(seeds, predicted, strict=True):
                seed_rows.append(
                    {"method_id": method, "seed": seed, "target": target, **numerical_metrics(truth, values)}
                )
            row = {
                "method_id": method,
                "target": target,
                "unit": unit,
                "n": len(events),
                "seed_count": len(seeds),
                "mae": float(means.mean()),
                "seed_mae_sd": float(means.std(ddof=1)) if len(seeds) > 1 else None,
                "rmse": float(np.sqrt(np.square(predicted - truth).mean(axis=1)).mean()),
                "bias": float((predicted - truth).mean()),
                "paired_mae_difference": float((errors - base).mean()),
                "negative_predictions": int((predicted < 0).sum()),
            }
            key = method + ":" + target
            intervals[key] = hierarchical_interval(errors - base, dates)
            group_metrics[key] = {}
            for mode in sorted({data.mode_cards[e]["mode_code"] for e in events}):
                mask = np.array([data.mode_cards[e]["mode_code"] == mode for e in events])
                group_metrics[key][mode] = {"n": int(mask.sum()), "mae": float(errors[:, mask].mean())}
            rows.append(row)
    return {
        "schema_version": "classical-evaluation.g6a.v1",
        "status": "completed" if len(events) == len(data.fold_for) else "completed_with_missing_predictions",
        "evaluation_mode": "historical_replay",
        "common_events": len(events),
        "common_event_hash": digest(events),
        "metrics": rows,
        "seed_metrics": seed_rows,
        "paired_ci95": intervals,
        "mode_metrics": group_metrics,
        "coverage": coverage,
        "new_method_count": len(CLASSICAL_METHODS),
        "registered_total": len(methods),
        "external_2026_read": False,
        "optimization_proxy_approval": False,
    }
