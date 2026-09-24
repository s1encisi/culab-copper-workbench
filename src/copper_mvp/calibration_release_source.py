"""Expose verified calibration studies through the existing artifact lifecycle."""

import json
from importlib import metadata
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from copper_mvp.calibration import CALIBRATION_VERSION, SplitC90
from copper_mvp.calibration_evaluation import read_calibration_training
from copper_mvp.common import WorkbenchError, digest, file_hash, safe
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_evaluation import paired_week_interval
from copper_mvp.model_registry import method_spec
from copper_mvp.model_training import safe_artifact_path, source_signature
from copper_mvp.release_contracts import TARGET_UNITS, TASK_ID


def load_calibrated_source(owner, actor, request):
    root = owner.source_root("calibration_study", request.source_id)
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    if state["status"] != "completed":
        raise WorkbenchError("校准研究尚未完成", "ARTIFACT_SOURCE_NOT_READY")
    training, protocol = read_calibration_training(root)
    if file_hash(root / "evaluation.json") != state["evaluation_sha256"]:
        raise WorkbenchError("校准评价记录哈希不一致", "MODEL_HASH_MISMATCH")
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    if source_signature(owner.data) != protocol["source"]:
        raise WorkbenchError("校准数据来源已变更", "SOURCE_CHANGED")
    available = protocol["expected_method_seeds"].get(request.method_id)
    if not available:
        raise WorkbenchError("校准研究没有该方法", "ARTIFACT_SOURCE")
    seed = available[0] if request.seed is None else request.seed
    if seed not in available:
        raise WorkbenchError("该种子不属于校准研究", "ARTIFACT_SOURCE")
    entries = [
        v
        for v in training["artifacts"]
        if v["method_id"] == request.method_id
        and v["seed"] == seed
        and v["status"] == "completed"
        and v["fold_id"] != "DEVELOPMENT"
    ]
    if {v["fold_id"] for v in entries} != set(owner.data.fold_for.values()):
        raise WorkbenchError("缺少完整外层校准工件", "ARTIFACT_SOURCE")
    splits = {v["fold_id"]: v for v in protocol["splits"]}
    artifacts = [
        {
            "fold_id": v["fold_id"],
            "path": v["model_path"],
            "sha256": v["model_sha256"],
            "fit_cutoff_at": v["calibration_fit_cutoff_at"],
            "base_fit_cutoff_at": v["base_fit_cutoff_at"],
            "kind": "calibrated_model",
            "train_event_hash": splits[v["fold_id"]]["fit_event_hash"],
            "calibration_event_hash": splits[v["fold_id"]]["calibration_event_hash"],
            "calibrator_path": v["calibrator_path"],
            "calibrator_sha256": v["calibrator_sha256"],
            "model_spec_hash": v["model_spec_hash"],
        }
        for v in entries
    ]
    frame = pd.read_csv(root / "predictions.csv")
    if frame.duplicated(["event_id", "method_id", "seed"]).any():
        raise WorkbenchError("校准预测记录重复", "ARTIFACT_SOURCE")
    common = set(owner.data.fold_for)
    by_case = {}
    for method, seeds in protocol["expected_method_seeds"].items():
        for current_seed in seeds:
            part = frame[frame.method_id.eq(method) & frame.seed.eq(current_seed)].set_index("event_id")
            needed = [
                "cu",
                "as",
                "marginal_c90_cu_lower",
                "marginal_c90_cu_upper",
                "marginal_c90_as_lower",
                "marginal_c90_as_upper",
            ]
            good = part.status.eq("completed") & np.isfinite(part[needed].to_numpy(float)).all(axis=1)
            common &= set(part.index[good])
            by_case[(method, current_seed)] = part
    labels = DataService(owner.data).labels.latest(protocol["evaluation_as_of"])
    common = {
        e
        for e in common
        if all(
            (e, t) in labels and labels[(e, t)].quality_eligible and labels[(e, t)].value is not None
            for t in ("cu", "as")
        )
    }
    events = sorted(common, key=lambda e: (owner.data.row(e).decision_at, e))
    if not events or digest(events) != evaluation["common_event_hash"]:
        raise WorkbenchError("校准比较的共同事件集合不一致", "ARTIFACT_SOURCE")
    target = request.target
    truth = np.array([labels[(e, target)].value for e in events])
    candidate = np.stack([by_case[(request.method_id, s)].loc[events, target].to_numpy(float) for s in available])
    base_seed = protocol["expected_method_seeds"]["Persistence"][0]
    baseline = by_case[("Persistence", base_seed)].loc[events, target].to_numpy(float)
    errors = np.abs(candidate - truth)
    delta = errors.mean(axis=0) - np.abs(baseline - truth)
    decisions = [source_time(owner.data.row(e).decision_at).isoformat() for e in events]
    timings = [
        t
        for row in training["artifacts"]
        if row["method_id"] == request.method_id and row["fold_id"] != "DEVELOPMENT"
        for t in row.get("single_prediction_ms", [])
    ]
    quality = {
        "mae": float(errors.mean()),
        "baseline_mae": float(np.abs(baseline - truth).mean()),
        "n": len(events),
        "paired_ci95": paired_week_interval(
            delta, decisions, {"minimum_blocks": 20, "replicates": 1000, "seed": protocol["request"]["seeds"][0]}
        ),
        "negative_predictions": int((candidate < 0).sum()),
        "coverage": min(v["fraction"] for v in evaluation["coverage"] if v["method_id"] == request.method_id),
        "p95_ms": float(np.percentile(timings, 95)) if timings else None,
        "mode_metrics": {},
        "baseline_modes": {},
        "seed_scope": (
            "All declared artifact seeds are retained in qualification; the sel"
            "ected seed only selects the saved estimator."
        ),
        "interval_metrics": [
            v for v in evaluation["metrics"] if v["method_id"] == request.method_id and v["target"] == target
        ],
        "joint_coverage": [v for v in evaluation["joint_coverage"] if v["method_id"] == request.method_id],
    }
    for mode in sorted({v["mode_code"] for v in owner.data.mode_cards.values()}):
        mask = np.array([owner.data.mode_cards[e]["mode_code"] == mode for e in events])
        if mask.any():
            quality["mode_metrics"][mode] = {"n": int(mask.sum()), "mae_mean_over_seeds": float(errors[:, mask].mean())}
            quality["baseline_modes"][mode] = {
                "n": int(mask.sum()),
                "mae_mean_over_seeds": float(np.abs(baseline[mask] - truth[mask]).mean()),
            }
    family = next(v for v in protocol["methods"][request.method_id] if v["seed"] == seed)
    if family["content_hash"] != method_spec(request.method_id, seed)["content_hash"]:
        raise WorkbenchError("校准模型的注册定义已变更", "MODEL_VERSION_MISMATCH")
    if request.method_id in ("CatBoost", "NGBoost", "EBM", "Cubist"):
        from copper_mvp.specialized_models import specialized_dependencies

        specialized_dependencies()
    dependencies = {name: metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")}
    dependencies.update({name: metadata.version(name) for name in family.get("dependencies", {})})
    bindings = {
        name: file_hash(root / name)
        for name in (
            "protocol.json",
            "training_manifest.json",
            "predictions.csv",
            "calibration_predictions.csv",
            "evaluation.json",
        )
    }
    descriptor = safe(
        {
            "schema_version": "model-artifact-set.g5c.v1",
            "source_kind": "calibration_study",
            "source_id": request.source_id,
            "method_id": request.method_id,
            "target": target,
            "unit": TARGET_UNITS[target],
            "seed": seed,
            "task_id": TASK_ID,
            "feature_columns": owner.data.feature_columns,
            "feature_spec_hash": digest(owner.data.feature_columns),
            "source": protocol["source"],
            "bindings": bindings,
            "artifacts": artifacts,
            "benchmark": quality,
            "family": family,
            "dependencies": dependencies,
            "license": {
                "project": "No new license assigned; see source project",
                "dependencies": dict.fromkeys(dependencies, "See installed distribution metadata"),
            },
            "calibrator": {
                "schema_version": CALIBRATION_VERSION,
                "nominal_coverage": protocol["request"]["coverage"],
                "modes": ["marginal_c90", "joint_c90"],
                "empirical_time_split": True,
                "unconditional_guarantee": False,
            },
            "calibrator_missing_reason": None,
            "scopes": ["historical_oof"],
            "proxy_approved": False,
            "causal_control": False,
            "evaluation_mode": "historical_replay",
            "benchmark_event_hash": evaluation["common_event_hash"],
        }
    )
    descriptor["source_fingerprint"] = digest(descriptor)
    owner.verify(descriptor)
    return descriptor


def predict_calibrated_source(owner, descriptor, event):
    fold = owner.data.fold_for.get(event)
    entry = next((v for v in descriptor["artifacts"] if v["fold_id"] == fold), None)
    if entry is None:
        raise WorkbenchError("该事件没有校准回放工件", "NO_OOF_MODEL")
    if source_time(entry["fit_cutoff_at"]) > source_time(owner.data.row(event).decision_at):
        raise WorkbenchError("校准器晚于事件决策时间", "FUTURE_MODEL")
    root = owner.source_root("calibration_study", descriptor["source_id"])
    for path_key, hash_key in (("path", "sha256"), ("calibrator_path", "calibrator_sha256")):
        if file_hash(safe_artifact_path(root, entry[path_key])) != entry[hash_key]:
            raise WorkbenchError("校准工件发生变化", "MODEL_HASH_MISMATCH")
    if entry["sha256"] not in owner.cache:
        owner.cache[entry["sha256"]] = joblib.load(root / entry["path"])
    model = owner.cache[entry["sha256"]]
    if model.spec["content_hash"] != entry["model_spec_hash"]:
        raise WorkbenchError("校准基础模型身份不一致", "MODEL_VERSION_MISMATCH")
    calibrator = SplitC90.from_manifest(json.loads((root / entry["calibrator_path"]).read_text(encoding="utf-8")))
    from threadpoolctl import threadpool_limits

    start = perf_counter()
    with threadpool_limits(limits=1):
        point = model.predict_context(owner.data, [event])
        intervals = calibrator.intervals(point)
    target_index = ("cu", "as").index(descriptor["target"])
    group_id = digest(
        {
            "study_id": descriptor["source_id"],
            "method_id": descriptor["method_id"],
            "seed": descriptor["seed"],
            "fold_id": fold,
            "calibrator": entry["calibrator_sha256"],
        }
    )
    evidence = {
        "artifact_sha256": entry["sha256"],
        "execution": "calibrated_model_inference",
        "model_elapsed_ms": (perf_counter() - start) * 1000,
        "calibrator_sha256": entry["calibrator_sha256"],
        "calibration": {
            "schema_version": CALIBRATION_VERSION,
            "nominal_coverage": calibrator.coverage,
            "group_id": group_id,
            "fit_cutoff_at": entry["fit_cutoff_at"],
            "base_fit_cutoff_at": entry["base_fit_cutoff_at"],
            "calibration_count": calibrator.calibration_count,
            "unconditional_guarantee": False,
        },
        "prediction_intervals": {
            mode: {
                "lower": float(value[0, 0, target_index]),
                "upper": float(value[0, 1, target_index]),
                "unit": descriptor["unit"],
                "joint_region": False,
                "projection_of_joint_region": mode == "joint_c90",
            }
            for mode, value in intervals.items()
        },
    }
    if "joint_c90" in intervals:
        bounds = intervals["joint_c90"]
        evidence["joint_prediction_region"] = {
            "id": digest({"group_id": group_id, "event_id": event}),
            "calibration_group_id": group_id,
            "nominal_coverage": calibrator.coverage,
            "bounds": {
                target: {"lower": float(bounds[0, 0, j]), "upper": float(bounds[0, 1, j]), "unit": TARGET_UNITS[target]}
                for j, target in enumerate(("cu", "as"))
            },
            "unconditional_guarantee": False,
        }
    return float(point[0, target_index]), evidence
