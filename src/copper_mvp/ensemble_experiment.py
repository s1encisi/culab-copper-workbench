"""Run and resume predeclared nested ensemble studies, then score independently."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from copper_mvp.common import WorkbenchError, digest, file_hash, utc_now, write_json
from copper_mvp.data_service import DataService
from copper_mvp.ensemble_evaluation import error_correlation, evaluate_ensembles, verify_fold
from copper_mvp.ensemble_models import BASE_METHODS, EnsembleSettings
from copper_mvp.ensemble_time import EnsembleData
from copper_mvp.ensemble_training import TrainingBudget, train_outer_fold
from copper_mvp.model_evaluation import read_training
from copper_mvp.model_training import source_signature

ENSEMBLE_FILES = (
    "ensemble_models.py",
    "ensemble_time.py",
    "ensemble_fusion.py",
    "ensemble_training.py",
    "ensemble_evaluation.py",
    "ensemble_experiment.py",
    "model_adapters.py",
    "modeling.py",
    "model_registry.py",
)
ENSEMBLE_METHODS = ("MeanEnsemble", "Stacking", "OOFConvex", "Blending", "DynamicConvex")
BAG_METHODS = ("Bagging_d1", "Bagging_d7", "BlockBagging")


def process_peak_mb():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if function(wintypes.HANDLE(-1), ctypes.byref(counter), counter.cb):
            return counter.PeakWorkingSetSize / 1024**2
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system() != "Darwin" else 1024**2)


def experiment_signature():
    return {
        "settings": EnsembleSettings().model_dump(),
        "code_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in ENSEMBLE_FILES},
    }


def run_ensemble_experiment(data, comparison_root, root, request, progress=lambda value: None):
    root, comparison_root = Path(root), Path(comparison_root)
    root.mkdir(parents=True, exist_ok=True)
    source_manifest, source_protocol = read_training(comparison_root)
    if source_manifest["status"] != "completed" or source_signature(data) != source_manifest["source"]:
        raise WorkbenchError("来源模型比较或数据版本不匹配", "ENSEMBLE_SOURCE")
    if not set(BASE_METHODS) <= set(source_protocol["request"]["methods"]):
        raise WorkbenchError("来源比较缺少所需基模型", "ENSEMBLE_SOURCE")
    settings = EnsembleSettings()
    signature = experiment_signature()
    ledger = DataService(data).labels
    book = EnsembleData(data, ledger)
    folds = []
    for old in source_protocol["folds"]:
        train = data.training_ids(old["fold_id"])
        valid = data.cv.loc[
            data.cv.fold_id.eq(old["fold_id"]) & data.cv.fold_role.eq("VALIDATION"), "origin_event_id"
        ].tolist()
        folds.append({**old, "train_event_ids": train, "validation_event_ids": valid})
    expected = {method: [-1] for method in source_protocol["request"]["methods"]}
    expected.update({method: [-1] for method in ENSEMBLE_METHODS})
    expected.update({method: list(request.seeds) for method in BAG_METHODS})
    protocol = {
        "schema_version": "ensemble-experiment.g5b.v1",
        "request": request.model_dump(mode="json"),
        "source": source_manifest["source"],
        "source_protocol_sha256": file_hash(comparison_root / "protocol.json"),
        "source_oof_sha256": source_manifest["oof_sha256"],
        "base_methods": BASE_METHODS,
        "base_seed": source_protocol["request"]["seed"],
        "settings": settings.model_dump(),
        "signature": signature,
        "folds": folds,
        "expected_methods": expected,
        "evaluation_as_of": source_protocol["evaluation_as_of"],
        "feature_columns": source_protocol["feature_columns"],
        "evaluation_mode": "historical_replay",
        "parameter_selection": "nested_forward_OOF_inside_each_original_outer_training_fold",
        "selection_objective": "per-target MAE; normalized joint MAE for bag member/block selection",
        "prediction_units": {"cu": "g/L", "as": "mg/L"},
        "shared_fit_accounting": (
            "Each fitted base or bag member is charged once; aliases and ablations share declared parents."
        ),
        "boosting_control": "Fixed DeltaHGB refit using the unchanged G2a specification.",
        "live_model_change": False,
        "optimization_proxy_approval": False,
        "causal_control": False,
        "external_2026_read": False,
        "method_inventory_increment": 0,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "native_threads": 1,
        },
    }
    if (root / "protocol.json").exists():
        previous = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        comparable = {k: v for k, v in previous.items() if k != "created_at"}
        if digest(comparable) != digest(protocol):
            raise WorkbenchError("恢复所需的数据、参数或代码已改变，请创建新研究", "SOURCE_CHANGED")
        protocol = previous
    else:
        protocol["created_at"] = utc_now()
        write_json(root / "protocol.json", protocol)
    budget = TrainingBudget(request.max_wall_seconds, progress)
    old_timings = (
        json.loads((root / "training_costs.json").read_text(encoding="utf-8"))
        if (root / "training_costs.json").exists()
        else []
    )
    results = []
    correlations = {}
    source_predictions = pd.read_csv(comparison_root / "oof_predictions.csv")
    try:
        with threadpool_limits(limits=1):
            for fold in folds:
                folder = root / "folds" / fold["fold_id"]
                if (folder / "fold_manifest.json").exists():
                    result = json.loads((folder / "fold_manifest.json").read_text(encoding="utf-8"))
                    verify_fold(folder, result, book, fold)
                    progress({"phase": "reuse_completed_fold", "fold_id": fold["fold_id"]})
                else:
                    progress({"phase": "nested_training", "fold_id": fold["fold_id"], "seed_count": len(request.seeds)})
                    result = train_outer_fold(
                        book, fold, settings, request.seeds, protocol["base_seed"], folder, budget
                    )
                    verify_fold(folder, result, book, fold)
                predictions = pd.read_csv(folder / "predictions.csv")
                for method in BASE_METHODS:
                    fresh = predictions[predictions.method_id.eq(method)].set_index("event_id")[["cu", "as"]]
                    original = (
                        source_predictions[source_predictions.method_id.eq(method)]
                        .set_index("event_id")
                        .loc[fresh.index, ["cu", "as"]]
                    )
                    if not np.allclose(fresh, original, rtol=1e-10, atol=1e-9):
                        raise WorkbenchError("重新拟合的原基线预测与 G2a 不一致", "ENSEMBLE_BASELINE_PARITY")
                inner = pd.read_csv(folder / "inner_oof.csv")
                actual = book.y(inner.event_id.tolist(), fold["fit_cutoff_at"])
                correlations[fold["fold_id"]] = {
                    target: error_correlation(
                        inner[[m + "_" + target for m in BASE_METHODS]].to_numpy(), actual[:, t], BASE_METHODS
                    )
                    for t, target in enumerate(("cu", "as"))
                }
                results.append({**result, "directory": folder.relative_to(root).as_posix()})
                write_json(root / "training_costs.json", old_timings + budget.fits)
                progress(
                    {
                        "phase": "fold_completed",
                        "fold_id": fold["fold_id"],
                        "completed_folds": len(results),
                        "fits_recorded": len(old_timings) + len(budget.fits),
                    }
                )
    finally:
        write_json(root / "training_costs.json", old_timings + budget.fits)
    if source_signature(data) != protocol["source"] or experiment_signature() != signature:
        raise WorkbenchError("训练期间数据或实现发生变化", "SOURCE_CHANGED")
    frames = []
    for result in results:
        frames.append(pd.read_csv(root / result["directory"] / "predictions.csv"))
    reference = source_predictions[~source_predictions.method_id.isin(BASE_METHODS)].copy()
    reference["seed"] = np.nan
    reference["fit_source"] = "G2a_existing_OOF"
    frames.append(reference)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(root / "predictions.csv", index=False)
    (root / "metric_snapshots").mkdir(exist_ok=True)
    with (root / "decisions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for result in results:
            with (root / result["directory"] / "weight_decisions.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    evidence = json.loads(line)
                    write_json(root / "metric_snapshots" / (evidence["id"] + ".json"), evidence)
                    decision = {
                        "event_id": evidence["event_id"],
                        "target": evidence["target"],
                        "method_id": "DynamicConvex",
                        "fold_id": result["fold_id"],
                        "metric_snapshot_id": evidence["id"],
                        "weights": evidence["weights"],
                        "as_of": evidence["as_of"],
                        "evaluation_mode": "historical_replay",
                    }
                    stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    fits = old_timings + budget.fits
    groups = {}
    for row in fits:
        group = "bootstrap" if row.get("bootstrap") else "meta" if row["method"] == "RidgeMeta" else "base"
        current = groups.setdefault(group, {"fits": 0, "fit_ms": 0.0, "failed": 0, "warnings": 0})
        current["fits"] += int(row.get("requires_fit", True))
        current["fit_ms"] += row["fit_ms"]
        current["failed"] += row["status"] == "failed"
        current["warnings"] += len(row.get("warnings", []))
    training_resources = {
        "fit_groups": groups,
        "all_recorded_fit_entries": len(fits),
        "actual_fit_calls": sum(bool(r.get("requires_fit", True)) for r in fits),
        "current_invocation_elapsed_ms": (time.perf_counter() - budget.start) * 1000,
        "current_invocation_cpu_seconds": time.process_time() - budget.cpu_start,
        "peak_process_rss_mb": process_peak_mb(),
        "memory_scope": "whole current study process, including loaded data and shared models",
        "full_training_cost_ledger_sha256": file_hash(root / "training_costs.json"),
        "reference_predictions_reused": reference.method_id.unique().tolist(),
        "llm_calls": 0,
        "tokens": 0,
        "api_cost_cny": 0,
    }
    manifest = {
        "schema_version": "ensemble-output.g5b.v1",
        "created_at": utc_now(),
        "source": protocol["source"],
        "protocol_sha256": file_hash(root / "protocol.json"),
        "predictions_sha256": file_hash(root / "predictions.csv"),
        "decisions_sha256": file_hash(root / "decisions.jsonl"),
        "folds": results,
        "training_resources": training_resources,
        "inner_residual_correlations": correlations,
        "baseline_refit_parity": {"rtol": 1e-10, "atol": 1e-9, "passed": True},
    }
    write_json(root / "replay_manifest.json", manifest)
    progress({"phase": "independent_evaluation"})
    result = evaluate_ensembles(data, root)
    progress({"phase": "completed", "common_events": result["common_events"]})
    return result
