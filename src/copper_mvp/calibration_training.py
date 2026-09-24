"""Fit models and C90 calibrators without accessing outer-validation labels."""

from __future__ import annotations

import math
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from copper_mvp.calibration import SplitC90, training_scale
from copper_mvp.common import APP_VERSION, PROJECT_ROOT, WorkbenchError, digest, file_hash, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_registry import method_spec
from copper_mvp.model_training import source_signature

# These fixed configurations do not use random state. Do not fabricate seed repeats.
DETERMINISTIC = {
    "Persistence",
    "DeltaRidge",
    "ElasticNet",
    "Huber",
    "PLS",
    "BayesianRidge",
    "ARD",
    "Quantile",
    "MultiTaskElasticNet",
    "KernelRidge",
    "SVR",
    "KNN",
    "GaussianProcess",
}
TARGETS = ("cu", "as")


def visible_labels(ledger, ids, cutoff):
    visible = ledger.latest(source_time(cutoff))
    rows = []
    for event in ids:
        pair = [visible.get((event, t)) for t in TARGETS]
        if any(
            v is None or not v.quality_eligible or v.value is None or v.available_at > source_time(cutoff) for v in pair
        ):
            raise WorkbenchError("校准使用的标签在截止时尚不可用", "CALIBRATION_LABEL_TIME")
        rows.extend(v.model_dump(mode="json") for v in pair)
    y = np.array([[visible[(event, t)].value for t in TARGETS] for event in ids], float)
    return y, digest(rows)


def temporal_split(data, ledger, outer_ids, valid_ids, cutoff, request, fold):
    ordered = sorted(outer_ids, key=lambda event: (source_time(data.row(event).decision_at), event))
    wanted = max(request.minimum_calibration, math.ceil(len(ordered) * request.calibration_fraction))
    if wanted >= len(ordered):
        raise WorkbenchError("该时间折不足以划分拟合与校准段", "CALIBRATION_SPLIT")
    start = source_time(data.row(ordered[-wanted]).decision_at)
    visible = ledger.latest(start)
    fit = []
    calibration = []
    purged = []
    for event in ordered:
        decision = source_time(data.row(event).decision_at)
        if decision >= start:
            calibration.append(event)
        elif all(
            (event, t) in visible
            and visible[(event, t)].quality_eligible
            and visible[(event, t)].value is not None
            and visible[(event, t)].available_at <= start
            for t in TARGETS
        ):
            fit.append(event)
        else:
            purged.append(event)
    if len(fit) < request.minimum_fit or len(calibration) < request.minimum_calibration:
        raise WorkbenchError("独立拟合段或校准段支持量不足", "INSUFFICIENT_CALIBRATION")
    if set(fit) & set(calibration) or (set(fit) | set(calibration)) & set(valid_ids):
        raise WorkbenchError("拟合、校准与验证事件相交", "CALIBRATION_SPLIT")
    if any(source_time(data.row(e).decision_at) >= source_time(cutoff) for e in calibration):
        raise WorkbenchError("校准事件越过外层截止", "CALIBRATION_SPLIT")
    if any(source_time(data.row(e).decision_at) < source_time(cutoff) for e in valid_ids):
        raise WorkbenchError("验证事件早于校准截止", "CALIBRATION_SPLIT")
    _, fit_labels_hash = visible_labels(ledger, fit, start)
    _, calibration_labels_hash = visible_labels(ledger, calibration, cutoff)
    return {
        "fold_id": fold,
        "base_fit_cutoff_at": start.isoformat(),
        "calibration_fit_cutoff_at": source_time(cutoff).isoformat(),
        "outer_ids": ordered,
        "outer_event_hash": digest(ordered),
        "fit_ids": fit,
        "calibration_ids": calibration,
        "validation_ids": list(valid_ids),
        "purged_ids": purged,
        "fit_event_hash": digest(fit),
        "calibration_event_hash": digest(calibration),
        "validation_event_hash": digest(valid_ids),
        "fit_labels_hash": fit_labels_hash,
        "calibration_labels_hash": calibration_labels_hash,
    }


def native_interval(model, data, ids):
    try:
        output = model.predict_uncertainty_context(data, ids)
    except WorkbenchError as error:
        if error.code in ("MODEL_UNCERTAINTY_UNSUPPORTED", "MODEL_CONTEXT_UNSUPPORTED"):
            return None
        raise
    if output["kind"] == "marginal_standard_deviation":
        mean = np.asarray(output["mean"], float)
        std = np.asarray(output["std"], float)
        if (
            mean.shape != (len(ids), 2)
            or std.shape != mean.shape
            or not np.isfinite([mean, std]).all()
            or (std <= 0).any()
        ):
            raise WorkbenchError("原生概率输出不合法", "CALIBRATION_NATIVE")
        return {
            "lower": mean - 1.6448536269514722 * std,
            "upper": mean + 1.6448536269514722 * std,
            "center": mean,
            "std": std,
            "coverage": 0.9,
            "kind": "native_normal_marginals",
        }
    if output["kind"] in ("raw_marginal_quantiles", "posterior_predictive_quantiles"):
        values = np.asarray(output["values"], float)
        levels = output["levels"]
        if values.shape != (len(ids), len(levels), 2) or not np.isfinite(values).all():
            raise WorkbenchError("原生分位数输出不合法", "CALIBRATION_NATIVE")
        return {
            "lower": values[:, 0, :],
            "upper": values[:, -1, :],
            "center": values[:, len(levels) // 2, :],
            "coverage": float(levels[-1] - levels[0]),
            "kind": "native_quantiles",
        }
    return None


def train_calibration_study(data, request, output, progress=lambda value: None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise WorkbenchError("校准目录已有协议，不能覆盖", "CALIBRATION_EXISTS")
    source = source_signature(data)
    ledger = DataService(data).labels
    splits = []
    for fold in sorted(data.cv.fold_id.unique()):
        rows = data.cv[data.cv.fold_id.eq(fold)]
        valid = rows.loc[rows.fold_role.eq("VALIDATION"), "origin_event_id"].tolist()
        splits.append(
            temporal_split(data, ledger, data.training_ids(fold), valid, rows.fold_fit_cutoff_at.iloc[0], request, fold)
        )
    full_cutoff = source_time(data.frame.decision_at.max())
    full_ids = [event for event in data.training_ids(None) if source_time(data.row(event).decision_at) < full_cutoff]
    # Only labels visible at the development cutoff enter the final calibration artifact.
    visible = ledger.latest(full_cutoff)
    full_ids = [
        e
        for e in full_ids
        if all(
            (e, t) in visible and visible[(e, t)].quality_eligible and visible[(e, t)].value is not None
            for t in TARGETS
        )
    ]
    splits.append(temporal_split(data, ledger, full_ids, [], full_cutoff, request, "DEVELOPMENT"))
    expected_seeds = {m: [request.seeds[0]] if m in DETERMINISTIC else request.seeds for m in request.methods}
    paths = sorted((PROJECT_ROOT / "src").rglob("*.py")) + sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    hashes = {path.relative_to(PROJECT_ROOT).as_posix(): file_hash(path) for path in paths}
    protocol = {
        "schema_version": "calibration-study.g6n.v1",
        "app_version": APP_VERSION,
        "request": request.model_dump(mode="json"),
        "source": source,
        "splits": splits,
        "expected_method_seeds": expected_seeds,
        "methods": {m: [method_spec(m, seed) for seed in seeds] for m, seeds in expected_seeds.items()},
        "evaluation_as_of": full_cutoff.isoformat(),
        "code_hashes": hashes,
        "created_at": utc_now(),
        "scale": "sample standard deviation of fit-segment target deltas; zero scale disables joint region",
        "independence": (
            "base preprocessing/parameters use fit segment only; calibration us"
            "es later mature labels; outer labels scored separately"
        ),
        "native_comparison": (
            "native uncertainty from the same reduced-fit model, not a model trained on calibration labels"
        ),
        "external_2026_read": False,
        "automatic_promotion": False,
        "latency_policy": {
            "single_event_samples": 32,
            "selection": "equally_spaced_outer_validation_events",
            "scope": "in_memory_base_model_plus_calibrator",
            "native_threads": 1,
        },
    }
    write_json(output / "protocol.json", protocol)
    for name in hashes:
        destination = output / "executed_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / name).read_bytes())
    predictions = []
    calibration_predictions = []
    artifacts = []
    started = perf_counter()

    def persist(status):
        for name, rows in (("predictions.csv", predictions), ("calibration_predictions.csv", calibration_predictions)):
            temporary = output / (name + ".tmp")
            columns = (
                None
                if rows
                else (
                    ["event_id", "fold_id", "method_id", "seed", "status", "decision_at", "cu", "as"]
                    if name == "predictions.csv"
                    else ["event_id", "fold_id", "method_id", "seed", "cu", "as"]
                )
            )
            pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
            temporary.replace(output / name)
        manifest = {
            "schema_version": "calibration-training.g6n.v1",
            "status": status,
            "source": source,
            "protocol_sha256": file_hash(output / "protocol.json"),
            "predictions_sha256": file_hash(output / "predictions.csv"),
            "calibration_predictions_sha256": file_hash(output / "calibration_predictions.csv"),
            "artifacts": artifacts,
            "elapsed_ms": (perf_counter() - started) * 1000,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "outer_validation_labels_used_during_training": False,
        }
        write_json(output / "training_manifest.json", manifest)
        return manifest

    with threadpool_limits(limits=1):
        for split in splits:
            fit_ids, cal_ids, valid_ids = split["fit_ids"], split["calibration_ids"], split["validation_ids"]
            fit_y, _ = visible_labels(ledger, fit_ids, split["base_fit_cutoff_at"])
            fit_X = data.X.loc[fit_ids].to_numpy(float)
            scale = training_scale(fit_X, fit_y)
            for method, seeds in expected_seeds.items():
                for seed in seeds:
                    if perf_counter() - started > request.max_wall_seconds:
                        persist("time_budget_exceeded")
                        raise WorkbenchError("校准研究达到时间预算", "CALIBRATION_TIME_BUDGET")
                    progress(
                        {"phase": "fit_and_calibrate", "fold_id": split["fold_id"], "method_id": method, "seed": seed}
                    )
                    entry = {
                        "fold_id": split["fold_id"],
                        "method_id": method,
                        "seed": seed,
                        "status": "running",
                        "fit_count": len(fit_ids),
                        "calibration_count": len(cal_ids),
                        "base_fit_cutoff_at": split["base_fit_cutoff_at"],
                        "calibration_fit_cutoff_at": split["calibration_fit_cutoff_at"],
                    }
                    fit_started = perf_counter()
                    clock = fit_started
                    try:
                        model = RegisteredModel(method, seed)
                        if model.spec["requires_fit"]:
                            if model.spec.get("input_kind") in ("raw_event_sequence", "anchored_process_sequence"):
                                model.fit_context(
                                    data,
                                    fit_ids,
                                    source_time(split["base_fit_cutoff_at"]),
                                    max_wall_seconds=request.max_wall_seconds - (perf_counter() - started),
                                )
                            else:
                                model.fit(
                                    fit_X, fit_y, max_wall_seconds=request.max_wall_seconds - (perf_counter() - started)
                                )
                        entry["base_fit_ms"] = (perf_counter() - clock) * 1000
                        if method == "BART" and not getattr(model, "fit_metadata", {}).get("diagnostics_passed", False):
                            raise WorkbenchError("后验采样诊断不合格，不能校准其输出", "SAMPLING_DIAGNOSTICS")
                        clock = perf_counter()
                        cal_point = model.predict_context(data, cal_ids)
                        entry["calibration_predict_ms"] = (perf_counter() - clock) * 1000
                        cal_y, _ = visible_labels(ledger, cal_ids, split["calibration_fit_cutoff_at"])
                        clock = perf_counter()
                        calibrator = SplitC90.fit(
                            cal_point, cal_y, scale, request.coverage, request.minimum_calibration
                        )
                        entry["calibrator_fit_ms"] = (perf_counter() - clock) * 1000
                        folder = output / "artifacts" / split["fold_id"] / method / str(seed)
                        folder.mkdir(parents=True, exist_ok=True)
                        model_path = folder / "model.joblib"
                        joblib.dump(model, model_path, compress=3)
                        cal_path = folder / "calibrator.json"
                        write_json(cal_path, calibrator.manifest())
                        entry.update(
                            status="completed",
                            model_path=model_path.relative_to(output).as_posix(),
                            model_sha256=file_hash(model_path),
                            calibrator_path=cal_path.relative_to(output).as_posix(),
                            calibrator_sha256=file_hash(cal_path),
                            joint_available=calibrator.joint_radius is not None,
                            warnings=model.fit_warnings,
                            model_spec_hash=model.spec["content_hash"],
                        )
                        for i, event in enumerate(cal_ids):
                            calibration_predictions.append(
                                {
                                    "event_id": event,
                                    "fold_id": split["fold_id"],
                                    "method_id": method,
                                    "seed": seed,
                                    "cu": cal_point[i, 0],
                                    "as": cal_point[i, 1],
                                }
                            )
                        if valid_ids:
                            clock = perf_counter()
                            point = model.predict_context(data, valid_ids)
                            entry["validation_predict_ms"] = (perf_counter() - clock) * 1000
                            bounds = calibrator.intervals(point)
                            samples = np.unique(np.linspace(0, len(valid_ids) - 1, min(32, len(valid_ids))).astype(int))
                            timings = []
                            for position in samples:
                                clock = perf_counter()
                                calibrator.intervals(model.predict_context(data, [valid_ids[position]]))
                                timings.append((perf_counter() - clock) * 1000)
                            entry["single_prediction_ms"] = timings
                            entry["single_p50_ms"] = float(np.percentile(timings, 50))
                            entry["single_p95_ms"] = float(np.percentile(timings, 95))
                            native = None
                            try:
                                native = native_interval(model, data, valid_ids)
                            except WorkbenchError as error:
                                entry["native_error"] = {"code": error.code, "message": str(error)}
                            for i, event in enumerate(valid_ids):
                                row = {
                                    "event_id": event,
                                    "fold_id": split["fold_id"],
                                    "method_id": method,
                                    "seed": seed,
                                    "status": "completed",
                                    "decision_at": source_time(data.row(event).decision_at).isoformat(),
                                    "base_fit_cutoff_at": split["base_fit_cutoff_at"],
                                    "calibration_fit_cutoff_at": split["calibration_fit_cutoff_at"],
                                    "cu": point[i, 0],
                                    "as": point[i, 1],
                                    "cu_scale": scale[0],
                                    "as_scale": scale[1],
                                }
                                for mode, values in bounds.items():
                                    for j, target in enumerate(TARGETS):
                                        row[mode + "_" + target + "_lower"] = values[i, 0, j]
                                        row[mode + "_" + target + "_upper"] = values[i, 1, j]
                                if native is not None:
                                    row.update(native_kind=native["kind"], native_coverage=native["coverage"])
                                    for j, target in enumerate(TARGETS):
                                        row["native_" + target + "_lower"] = native["lower"][i, j]
                                        row["native_" + target + "_upper"] = native["upper"][i, j]
                                        row["native_" + target + "_center"] = native["center"][i, j]
                                        if "std" in native:
                                            row["native_" + target + "_std"] = native["std"][i, j]
                                predictions.append(row)
                    except Exception as error:
                        entry.update(
                            status="failed",
                            error={"code": getattr(error, "code", type(error).__name__), "message": str(error)[:600]},
                        )
                        for event in valid_ids:
                            predictions.append(
                                {
                                    "event_id": event,
                                    "fold_id": split["fold_id"],
                                    "method_id": method,
                                    "seed": seed,
                                    "status": "failed",
                                    "decision_at": source_time(data.row(event).decision_at).isoformat(),
                                }
                            )
                    entry["total_ms"] = (perf_counter() - fit_started) * 1000
                    artifacts.append(entry)
                    persist("running")
    if source_signature(data) != source or any(
        file_hash(PROJECT_ROOT / name) != value for name, value in hashes.items()
    ):
        persist("source_changed")
        raise WorkbenchError("校准期间源文件发生变化", "SOURCE_CHANGED")
    return persist("completed")
