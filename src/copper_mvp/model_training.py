"""Train fixed G2a methods; persist OOF predictions without scoring validation labels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from copper_mvp.common import APP_VERSION, WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_registry import ComparisonRequest, method_spec


def source_signature(data):
    hashes = {key: file_hash(path) for key, path in data.paths.sources().items()}
    return {"dataset_version": data.dataset_version, "source_hashes": hashes, "signature": digest(hashes)}


def safe_artifact_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise WorkbenchError("工件路径越界", "ARTIFACT_PATH")
    return path


def train_comparison(data, request: ComparisonRequest, output: Path, progress=lambda detail: None):
    output = Path(output)
    if (output / "training_manifest.json").exists() or (output / "protocol.json").exists():
        raise WorkbenchError("该比较目录已有训练记录", "COMPARISON_EXISTS")
    output.mkdir(parents=True, exist_ok=True)
    source = source_signature(data)
    service = DataService(data)
    ledger = service.labels
    folds = []
    all_validation = []
    for fold in sorted(data.cv.fold_id.unique()):
        rows = data.cv[data.cv.fold_id.eq(fold)]
        train_ids = data.training_ids(fold)
        valid_ids = rows.loc[rows.fold_role.eq("VALIDATION"), "origin_event_id"].tolist()
        cutoff = source_time(rows.fold_fit_cutoff_at.iloc[0])
        if not train_ids or not valid_ids or set(train_ids) & set(valid_ids):
            raise WorkbenchError("训练与验证集合为空或相交", "COMPARISON_SPLIT")
        if (any(source_time(data.row(e).decision_at) >= cutoff for e in train_ids)
            or any(source_time(data.row(e).decision_at) < cutoff for e in valid_ids)):
            raise WorkbenchError("比较时间折越过决策截止", "TEMPORAL_SPLIT")
        folds.append({"fold_id": fold, "fit_cutoff_at": cutoff.isoformat(), "train_count": len(train_ids),
                      "validation_count": len(valid_ids), "train_event_hash": digest(train_ids),
                      "validation_event_hash": digest(valid_ids)})
        all_validation.extend(valid_ids)
    if len(set(all_validation)) != len(all_validation) or set(all_validation) != set(data.fold_for):
        raise WorkbenchError("OOF 事件集合不一致", "COMPARISON_SPLIT")
    full_cutoff = source_time(data.frame.decision_at.max())
    protocol = {
        "schema_version": "comparison-protocol.g2a.v1", "app_version": APP_VERSION, "evaluation_mode": "historical_replay",
        "created_at": utc_now(), "request": request.model_dump(mode="json"), **source,
        "task_id": "copper-next-recorded-result", "feature_columns": data.feature_columns,
        "methods": [method_spec(m, request.seed) for m in request.methods], "folds": folds,
        "evaluation_as_of": full_cutoff.isoformat(), "expected_oof_events": len(all_validation),
        "parameter_policy": "fixed_before_evaluation_no_search", "seed_policy": "one_predeclared_seed_all_fold_results_retained",
        "comparison_cohort": "common_finite_two_target_predictions_and_mature_eligible_labels",
        "uncertainty": {"method": "paired_calendar_week_block_bootstrap", "replicates": 1000,
                        "confidence": 0.95, "minimum_blocks": 20, "seed": request.seed},
        "latency_policy": {"scope": "warm_single_row_two_target_prediction", "samples_per_fold": 32,
                           "selection": "evenly_spaced_validation_positions", "native_threads": 1},
        "automatic_promotion": False, "optimization_proxy_approval": False, "external_2026_read": False,
        "code_hashes": {name: file_hash(Path(__file__).with_name(name))
                        for name in ("model_adapters.py", "model_registry.py", "model_training.py")},
    }
    if any(m not in ("Persistence", "DeltaRidge", "DeltaHGB", "ElasticNet", "Huber", "PLS") for m in request.methods):
        protocol["code_hashes"].update({name: file_hash(Path(__file__).with_name(name)) for name in ("classical_registry.py", "classical_models.py", "classical_evaluation.py")})
    if any(method_spec(m, request.seed).get("input_kind") in ("raw_event_sequence", "current_result_pair") for m in request.methods):
        protocol["code_hashes"].update({name: file_hash(Path(__file__).with_name(name)) for name in ("statistical_registry.py", "statistical_models.py", "statistical_runtime.py")})
        protocol["latency_policy"]["event_context_methods"] = {"methods": [m for m in request.methods if method_spec(m, request.seed).get("input_kind") == "raw_event_sequence"],
            "samples_per_fold": 8, "scope": "loaded_parameters_with_causal_event_context_reconstruction", "mean_and_std_computed_together": True}
    if any(m in ("CatBoost", "NGBoost", "EBM", "Cubist") for m in request.methods):
        protocol["code_hashes"].update({name: file_hash(Path(__file__).with_name(name)) for name in ("specialized_registry.py", "specialized_models.py", "specialized_evaluation.py", "cubist_model.py")})
    if "BART" in request.methods:
        from copper_mvp.bart_registry import bart_source_hashes
        protocol["code_hashes"].update(bart_source_hashes())
    if "SymbolicRegression" in request.methods:
        from copper_mvp.symbolic_registry import symbolic_source_hashes
        protocol["code_hashes"].update(symbolic_source_hashes())
    if any(m in ("TabNet", "FTTransformer", "NODE") for m in request.methods):
        from copper_mvp.tabular_registry import tabular_source_hashes
        protocol["code_hashes"].update(tabular_source_hashes())
    if any(method_spec(m, request.seed).get("input_kind") == "anchored_process_sequence" for m in request.methods):
        from copper_mvp.temporal_registry import temporal_source_hashes
        protocol["code_hashes"].update(temporal_source_hashes())
    write_json(output / "protocol.json", protocol)
    started = time.perf_counter()
    timings = []; artifacts = []; predictions = []; uncertainty_predictions = []
    scopes = [(f["fold_id"], data.training_ids(f["fold_id"]),
               data.cv.loc[data.cv.fold_id.eq(f["fold_id"]) & data.cv.fold_role.eq("VALIDATION"), "origin_event_id"].tolist(),
               source_time(f["fit_cutoff_at"])) for f in folds]
    scopes.append(("DEVELOPMENT", data.training_ids(None), [], full_cutoff))

    runtime = {}
    def persist(status):
        columns = ["event_id", "fold_id", "method_id", "decision_at", "fit_cutoff_at", "computed_at", "cu", "as", "status"]
        temporary = output / "oof_predictions.csv.tmp"
        pd.DataFrame(predictions, columns=columns).to_csv(temporary, index=False)
        temporary.replace(output / "oof_predictions.csv")
        uncertainty_hash = None
        if uncertainty_predictions:
            pd.DataFrame(uncertainty_predictions).to_csv(output / "oof_uncertainty.csv", index=False)
            uncertainty_hash = file_hash(output / "oof_uncertainty.csv")
        current = safe({"schema_version": "comparison-training.g2a.v1", "status": status,
            "created_at": utc_now(), "evaluation_mode": "historical_replay", "source": source,
            "protocol_sha256": file_hash(output / "protocol.json"), "oof_sha256": file_hash(output / "oof_predictions.csv"),
            "artifacts": artifacts, "timings": timings, "runtime": runtime, "uncertainty_sha256": uncertainty_hash,
            "elapsed_ms": (time.perf_counter() - started) * 1000, "automatic_promotion": False})
        write_json(output / "training_manifest.json", current)
        return current

    with threadpool_limits(limits=1):
        runtime = {"python": platform.python_version(), "platform": platform.platform(),
                   "logical_cpus": os.cpu_count(), "threadpools": [{k: v for k, v in item.items() if k != "filepath"} for item in threadpool_info()]}
        for fold, train_ids, valid_ids, cutoff in scopes:
            visible = ledger.latest(cutoff)
            if not all((e, t) in visible and visible[(e, t)].quality_eligible
                       and visible[(e, t)].value is not None for e in train_ids for t in ("cu", "as")):
                raise WorkbenchError("训练标签在截止时尚未成熟或无效", "IMMATURE_TRAINING_LABEL")
            X_train = data.X.loc[train_ids].to_numpy(float)
            y_train = np.array([[visible[(e, t)].value for t in ("cu", "as")] for e in train_ids])
            X_valid = data.X.loc[valid_ids].to_numpy(float) if valid_ids else None
            for method_id in request.methods:
                if time.perf_counter() - started > request.max_wall_seconds:
                    persist("time_budget_exceeded")
                    raise WorkbenchError("比较到达阶段边界时间预算", "COMPARISON_TIME_BUDGET")
                progress({"phase": "training", "fold_id": fold, "method_id": method_id})
                model = RegisteredModel(method_id, request.seed)
                timing = {"fold_id": fold, "method_id": method_id, "train_events": len(train_ids),
                          "status": "completed", "fit_ms": 0.0, "batch_predict_ms": 0.0,
                          "single_prediction_ms": [], "warnings": [], "retries": 0, "cache_hits": 0}
                entry = {"fold_id": fold, "method_id": method_id, "fit_cutoff_at": cutoff.isoformat(),
                         "scope": "development_analysis" if fold == "DEVELOPMENT" else "oof_replay",
                         "train_count": len(train_ids), "train_event_hash": digest(train_ids),
                         "requires_fit": model.spec["requires_fit"], "spec_hash": model.spec["content_hash"]}
                try:
                    if model.spec["requires_fit"]:
                        clock = time.perf_counter()
                        try:
                            if model.spec.get("input_kind") == "anchored_process_sequence":
                                model.fit_context(data, train_ids, cutoff, max_wall_seconds=request.max_wall_seconds-(time.perf_counter()-started))
                            elif model.spec.get("input_kind") == "raw_event_sequence":
                                model.fit_context(data, train_ids, cutoff)
                            else:
                                if method_id == "CatBoost":
                                    ordered_times = [source_time(data.row(event).decision_at) for event in train_ids]
                                    if any(a > b for a, b in zip(ordered_times, ordered_times[1:])):
                                        raise WorkbenchError("CatBoost 训练行必须按时间排序", "MODEL_TRAINING_ORDER")
                                if method_id in ("BART", "SymbolicRegression", "TabNet", "FTTransformer", "NODE"):
                                    model.fit(X_train, y_train, max_wall_seconds=request.max_wall_seconds-(time.perf_counter()-started))
                                else:
                                    model.fit(X_train, y_train)
                                if method_id == "CatBoost":
                                    model.fit_metadata["chronological_training_verified"] = True
                        finally:
                            timing["fit_ms"] = (time.perf_counter() - clock) * 1000
                        timing["warnings"] = model.fit_warnings
                        if hasattr(model, "fit_metadata"):
                            entry["fit_metadata"] = model.fit_metadata
                    folder = output / "artifacts" / fold / method_id
                    folder.mkdir(parents=True, exist_ok=True)
                    if model.spec["requires_fit"]:
                        path = folder / "model.joblib"
                        temp = folder / "model.joblib.tmp"
                        joblib.dump(model, temp, compress=3)
                        temp.replace(path)
                    else:
                        path = folder / "reference.json"
                        write_json(path, {"method_id": method_id, "requires_fit": False, "spec": model.spec})
                    entry.update({"path": path.relative_to(output).as_posix(), "sha256": file_hash(path), "status": "completed"})
                    if valid_ids:
                        clock = time.perf_counter()
                        values = model.predict_context(data, valid_ids)
                        timing["batch_predict_ms"] = (time.perf_counter() - clock) * 1000
                        if model.spec.get("capabilities", {}).get("uncertainty") in ("marginal_std", "raw_quantiles"):
                            clock = time.perf_counter()
                            distribution = model.predict_uncertainty_context(data, valid_ids)
                            timing["uncertainty_predict_ms"] = (time.perf_counter() - clock) * 1000
                            for i, event in enumerate(valid_ids):
                                for t, target in enumerate(("cu", "as")):
                                    row = {"event_id": event, "fold_id": fold, "method_id": method_id, "target": target,
                                           "kind": distribution["kind"], "prediction": float(values[i, t])}
                                    if distribution["kind"] == "marginal_standard_deviation":
                                        row["std"] = float(distribution["std"][i, t])
                                    else:
                                        row.update({"q10": float(distribution["values"][i, 0, t]), "q50": float(distribution["values"][i, 1, t]),
                                                    "q90": float(distribution["values"][i, 2, t])})
                                    uncertainty_predictions.append(row)
                        contextual = model.spec.get("input_kind") in ("raw_event_sequence", "anchored_process_sequence")
                        timing["prediction_scope"] = "event_context_reconstruction" if contextual else "single_row_matrix_inference"
                        positions = np.unique(np.linspace(0, len(valid_ids) - 1, min(8 if contextual else 32, len(valid_ids))).astype(int))
                        for position in positions:
                            clock = time.perf_counter()
                            model.predict_context(data, [valid_ids[position]])
                            timing["single_prediction_ms"].append((time.perf_counter() - clock) * 1000)
                        for event, value in zip(valid_ids, values):
                            predictions.append({"event_id": event, "fold_id": fold, "method_id": method_id,
                                "decision_at": source_time(data.row(event).decision_at).isoformat(),
                                "fit_cutoff_at": cutoff.isoformat(), "computed_at": utc_now(),
                                "cu": float(value[0]), "as": float(value[1]), "status": "completed"})
                except Exception as exc:
                    code = getattr(exc, "code", type(exc).__name__)
                    timing.update({"status": "failed", "error_code": code})
                    entry.update({"status": "failed", "error_code": code})
                    for event in valid_ids:
                        predictions.append({"event_id": event, "fold_id": fold, "method_id": method_id,
                            "decision_at": source_time(data.row(event).decision_at).isoformat(),
                            "fit_cutoff_at": cutoff.isoformat(), "computed_at": utc_now(),
                            "cu": None, "as": None, "status": "failed"})
                artifacts.append(entry); timings.append(timing)
                persist("running")
                progress({"phase": "trained", "fold_id": fold, "method_id": method_id,
                          "status": timing["status"], "fit_ms": round(timing["fit_ms"], 2),
                          "warnings": len(timing["warnings"])})
    if source_signature(data) != source:
        raise WorkbenchError("比较期间数据来源发生变化", "SOURCE_CHANGED")
    return persist("completed" if all(a["status"] == "completed" for a in artifacts) else "partial_failure")


def load_registered_model(root: Path, manifest: dict, protocol: dict, method_id: str, fold_id: str):
    entry = next((a for a in manifest["artifacts"] if a["method_id"] == method_id and a["fold_id"] == fold_id), None)
    if entry is None or entry["status"] != "completed":
        raise WorkbenchError("该比较没有可用模型工件", "COMPARISON_MODEL_NOT_FOUND")
    recorded = next((m for m in protocol["methods"] if m["method_id"] == method_id), None)
    current = method_spec(method_id, protocol["request"]["seed"])
    if (recorded is None or entry["spec_hash"] != recorded["content_hash"]
        or recorded["implementation"] != current["implementation"]
        or recorded["package_version"] != current["package_version"]):
        raise WorkbenchError("方法实现/预设与模型工件不一致", "COMPARISON_MODEL_VERSION")
    if method_id in ("CatBoost", "NGBoost", "EBM", "Cubist"):
        from copper_mvp.specialized_models import specialized_dependencies
        specialized_dependencies()
        if recorded["method_version"] != current["method_version"]:
            raise WorkbenchError("专用模型实现版本不兼容", "COMPARISON_MODEL_VERSION")
    if recorded.get("input_kind") in ("raw_event_sequence", "current_result_pair"):
        from copper_mvp.statistical_runtime import statistical_dependencies
        statistical_dependencies()
        if recorded["method_version"] != current["method_version"]:
            raise WorkbenchError("统计模型状态实现版本不兼容，请使用重新验证的工件", "COMPARISON_MODEL_VERSION")
    if method_id == "BART":
        from copper_mvp.bart_trees import numeric_evaluator
        numeric_evaluator()
        if recorded["method_version"] != current["method_version"]:
            raise WorkbenchError("BART 模型实现版本不兼容", "COMPARISON_MODEL_VERSION")
    if method_id == "SymbolicRegression" and recorded["method_version"] != current["method_version"]:
        raise WorkbenchError("符号回归实现版本不兼容", "COMPARISON_MODEL_VERSION")
    if method_id in ("TabNet", "FTTransformer", "NODE"):
        from copper_mvp.tabular_runtime import tabular_runtime
        tabular_runtime()
        if recorded["method_version"] != current["method_version"]:
            raise WorkbenchError("表格神经模型版本不兼容", "COMPARISON_MODEL_VERSION")
    if recorded.get("input_kind") == "anchored_process_sequence":
        from copper_mvp.neural_runtime import load_tensor_runtime
        load_tensor_runtime()
        if recorded["method_version"] != current["method_version"]:
            raise WorkbenchError("时序神经模型版本不兼容", "COMPARISON_MODEL_VERSION")
    path = safe_artifact_path(root, entry["path"])
    if not path.is_file() or file_hash(path) != entry["sha256"]:
        raise WorkbenchError("比较模型工件哈希不匹配", "MODEL_HASH_MISMATCH")
    model = joblib.load(path) if entry["requires_fit"] else RegisteredModel(method_id, protocol["request"]["seed"])
    if not isinstance(model, RegisteredModel) or model.method_id != method_id or model.seed != protocol["request"]["seed"] or model.spec["content_hash"] != entry["spec_hash"]:
        raise WorkbenchError("模型工件身份与注册记录不符", "COMPARISON_MODEL_VERSION")
    return model, entry
