"""Nested ensemble fitting with explicit temporal lineage and shared fit accounting."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from copper_mvp.common import WorkbenchError, digest, file_hash, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.ensemble_fusion import AdaptiveTarget, choose_meta, meta_features
from copper_mvp.ensemble_models import (
    BASE_METHODS,
    BaggedPredictor,
    ConvexPredictor,
    StackedPredictor,
    expand_weights,
    make_meta,
    simplex_weights,
    training_scale,
)
from copper_mvp.ensemble_time import block_sample, select_with_tolerance
from copper_mvp.model_adapters import RegisteredModel


class TrainingBudget:
    def __init__(self, max_seconds, progress):
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.max_seconds = max_seconds
        self.progress = progress
        self.fits = []

    def check(self):
        if time.perf_counter() - self.start > self.max_seconds:
            raise WorkbenchError("集成训练已到达时间预算，已完成的折与工件保留", "ENSEMBLE_TIME_BUDGET")

    def fit_base(self, method, seed, X, y, scope, extra=None):
        self.check()
        start = time.perf_counter()
        entry = {
            "method": method,
            "seed": seed,
            "scope": scope,
            "train_events": len(X),
            "status": "running",
            **(extra or {}),
        }
        model = RegisteredModel(method, seed)
        try:
            if model.spec["requires_fit"]:
                model.fit(X, y)
            entry.update(status="completed", warnings=model.fit_warnings, requires_fit=model.spec["requires_fit"])
            return model
        except Exception as exc:
            entry.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__))
            raise
        finally:
            entry["fit_ms"] = (time.perf_counter() - start) * 1000
            self.fits.append(entry)
            if len(self.fits) % 20 == 0:
                self.progress({"phase": "training", "scope": scope, "fits_recorded": len(self.fits)})


def predict_bases(models, X):
    return np.stack([model.predict(X) for model in models], axis=1)


def save_predictor(root, name, predictor, X_valid, budget, metadata, seed=None):
    budget.check()
    folder = root / "artifacts" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.joblib"
    temporary = folder / "model.joblib.tmp"
    joblib.dump(predictor, temporary, compress=3)
    temporary.replace(path)
    start = time.perf_counter()
    values = predictor.predict(X_valid)
    batch_ms = (time.perf_counter() - start) * 1000
    positions = np.unique(np.linspace(0, len(X_valid) - 1, min(16, len(X_valid))).astype(int))
    samples = []
    for position in positions:
        start = time.perf_counter()
        predictor.predict(X_valid[position : position + 1])
        samples.append((time.perf_counter() - start) * 1000)
    entry = {
        "name": name,
        "seed": seed,
        "path": path.relative_to(root).as_posix(),
        "sha256": file_hash(path),
        "artifact_type": "static_predictor",
        "batch_predict_ms": batch_ms,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "latency_samples_ms": samples,
        "timing_samples": len(samples),
        "timing_scope": "warm_single_event_two_target_inference",
        **metadata,
    }
    return values, entry


def train_outer_fold(book, fold, settings, seeds, base_seed, root, budget):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_ids = sorted(fold["train_event_ids"], key=lambda e: (book.decision(e), e))
    valid_ids = sorted(fold["validation_event_ids"], key=lambda e: (book.decision(e), e))
    cutoff = source_time(fold["fit_cutoff_at"])
    if set(train_ids) & set(valid_ids) or any(book.decision(e) < cutoff for e in valid_ids):
        raise WorkbenchError("外层训练与验证边界不成立", "ENSEMBLE_OUTER_SPLIT")
    plan = book.inner_plan(train_ids, cutoff, settings)
    write_json(
        root / "inner_plan.json",
        {
            "outer_fold": fold["fold_id"],
            "outer_fit": book.evidence(train_ids, cutoff),
            "outer_train_ids": train_ids,
            "outer_validation_ids": valid_ids,
            "inner_folds": plan,
        },
    )
    positions = {e: i for i, e in enumerate(train_ids)}
    X = book.X(train_ids)
    Y = book.y(train_ids, cutoff)
    X_valid = book.X(valid_ids)
    P = np.full((len(train_ids), 3, 2), np.nan)
    inner_data = []
    last_models = None
    for part in plan:
        at = source_time(part["fit_cutoff_at"])
        ids = part["train_event_ids"]
        ids_valid = part["validation_event_ids"]
        Xt, yt, Xv = book.X(ids), book.y(ids, at), book.X(ids_valid)
        scope = fold["fold_id"] + "/inner-" + str(part["inner_fold"])
        models = [
            budget.fit_base(m, base_seed, Xt, yt, scope, {"lineage": book.evidence(ids, at)}) for m in BASE_METHODS
        ]
        values = predict_bases(models, Xv)
        indices = [positions[e] for e in ids_valid]
        P[indices] = values
        inner_data.append((part, Xt, yt, Xv, indices))
        last_models = models
    oof_positions = np.flatnonzero(np.isfinite(P).all(axis=(1, 2)))
    oof_ids = [train_ids[i] for i in oof_positions]
    Z = P[oof_positions]
    current = X[oof_positions, :2]
    oof_values = book.y(oof_ids, cutoff)
    columns = {"event_id": oof_ids}
    for j, method in enumerate(BASE_METHODS):
        for t, target in enumerate(("cu", "as")):
            columns[method + "_" + target] = Z[:, j, t]
    pd.DataFrame(columns).to_csv(root / "inner_oof.csv", index=False)
    meta_cut = source_time(plan[-1]["fit_cutoff_at"])
    meta_train = book.mature([e for e in oof_ids if book.decision(e) < meta_cut], meta_cut)
    meta_valid = plan[-1]["validation_event_ids"]
    if len(meta_train) < 20:
        raise WorkbenchError("二层训练缺少向前 OOF 样本", "ENSEMBLE_META_SUPPORT")
    scale, scale_method = training_scale(inner_data[0][2])
    selection = choose_meta(book, Z, current, oof_ids, meta_train, meta_valid, meta_cut, cutoff, settings, scale)
    selection.update(
        normalization_scale=scale.tolist(),
        normalization_source=scale_method,
        final_meta_fit=book.evidence(oof_ids, cutoff),
        warmup_events_not_used_for_meta=len(train_ids) - len(oof_ids),
    )
    write_json(root / "selection.json", selection)
    original_order = fold["train_event_ids"]
    full_models = [
        budget.fit_base(
            m,
            base_seed,
            book.X(original_order),
            book.y(original_order, cutoff),
            fold["fold_id"] + "/full",
            {"lineage": book.evidence(original_order, cutoff)},
        )
        for m in BASE_METHODS
    ]
    base_values = predict_bases(full_models, X_valid)
    predictions = {}
    artifacts = []
    weight_decisions = []
    metadata = {
        "fold_id": fold["fold_id"],
        "fit_cutoff_at": cutoff.isoformat(),
        "feature_count": X.shape[1],
        "train_event_hash": digest(train_ids),
        "inner_plan_sha256": file_hash(root / "inner_plan.json"),
        "selection_sha256": file_hash(root / "selection.json"),
    }
    for j, method in enumerate(BASE_METHODS):
        predictions[(method, None)] = base_values[:, j, :]
    meta_models = []
    subsets = []
    for target, spec in enumerate(selection["selected"]["stacking"]):
        subset = spec["subset"]
        meta = make_meta(spec["alpha"])
        start = time.perf_counter()
        meta.fit(meta_features(Z, current, subset, target), oof_values[:, target] - current[:, target])
        budget.fits.append(
            {
                "method": "RidgeMeta",
                "scope": fold["fold_id"] + "/meta",
                "target": target,
                "fit_ms": (time.perf_counter() - start) * 1000,
                "train_events": len(oof_ids),
                "status": "completed",
                "requires_fit": True,
                "lineage": book.evidence(oof_ids, cutoff),
            }
        )
        meta_models.append(meta)
        subsets.append(subset)
    static_weights = []
    blend_weights = []
    initial_weights = []
    tail_positions = [positions[e] for e in meta_valid]
    tail_y = book.y(meta_valid, cutoff)
    for target in (0, 1):
        spec = selection["selected"]["convex"][target]
        subset = spec["subset"]
        static_weights.append(
            expand_weights(
                simplex_weights(Z[:, subset, target], oof_values[:, target], scale[target], spec["penalty"]), subset
            )
        )
        blend_weights.append(
            expand_weights(
                simplex_weights(
                    P[tail_positions][:, subset, target], tail_y[:, target], scale[target], spec["penalty"]
                ),
                subset,
            )
        )
        spec = selection["selected"]["adaptive"][target]
        subset = spec["subset"]
        initial_weights.append(
            expand_weights(
                simplex_weights(Z[:, subset, target], oof_values[:, target], scale[target], spec["penalty"]), subset
            )
        )
    estimators = {
        "Stacking": StackedPredictor(full_models, meta_models, subsets),
        "OOFConvex": ConvexPredictor(full_models, static_weights),
        "Blending": ConvexPredictor(last_models, blend_weights),
        "MeanEnsemble": ConvexPredictor(full_models, np.full((2, 3), 1 / 3)),
    }
    for name, model in estimators.items():
        lineage = {
            **metadata,
            "base_fit_cutoff_at": plan[-1]["fit_cutoff_at"] if name == "Blending" else cutoff.isoformat(),
            "meta_training_event_hash": digest(meta_valid if name == "Blending" else oof_ids)
            if name != "MeanEnsemble"
            else None,
        }
        values, artifact = save_predictor(root, name, model, X_valid, budget, lineage)
        predictions[(name, None)] = values
        artifacts.append(artifact)
    dynamic_config = {
        "parameters": selection["selected"]["adaptive"],
        "initial_weights": np.array(initial_weights).tolist(),
        "scale": scale.tolist(),
        "base_methods": BASE_METHODS,
        "base_seed": base_seed,
        "initial_history_event_ids": oof_ids,
        "initial_history_predictions": Z.tolist(),
        "settings": settings.model_dump(),
        **metadata,
    }
    path = root / "artifacts" / "DynamicConvex" / "state.json"
    write_json(path, dynamic_config)
    initial_state_hash = file_hash(path)
    mixers = [
        AdaptiveTarget(
            book,
            t,
            selection["selected"]["adaptive"][t],
            initial_weights[t],
            scale[t],
            [(e, Z[i, :, t]) for i, e in enumerate(oof_ids)],
            settings,
        )
        for t in (0, 1)
    ]
    dynamic_values = []
    dynamic_times = []
    for i, event in enumerate(valid_ids):
        if i % 100 == 0:
            budget.check()
        start = time.perf_counter()
        row = []
        for target, mixer in enumerate(mixers):
            value, evidence = mixer.predict(event, base_values[i, :, target])
            evidence.update(
                fold_id=fold["fold_id"],
                base_fit_cutoff_at=cutoff.isoformat(),
                base_forecast_hash=digest(base_values[i].tolist()),
                initial_state_sha256=initial_state_hash,
            )
            evidence.pop("id")
            evidence["id"] = digest(evidence)
            weight_decisions.append(evidence)
            row.append(value)
        dynamic_values.append(row)
        dynamic_times.append((time.perf_counter() - start) * 1000)
    if file_hash(path) != initial_state_hash:
        raise WorkbenchError("动态融合初始状态在回放期间改变", "SOURCE_CHANGED")
    predictions[("DynamicConvex", None)] = np.array(dynamic_values)
    artifacts.append(
        {
            "name": "DynamicConvex",
            "seed": None,
            "path": path.relative_to(root).as_posix(),
            "sha256": file_hash(path),
            "artifact_type": "stateful_replay",
            "p50_ms": float(np.percentile(dynamic_times, 50)),
            "p95_ms": float(np.percentile(dynamic_times, 95)),
            "latency_samples_ms": dynamic_times,
            "timing_samples": len(dynamic_times),
            "timing_scope": "two-target weight update and evidence; base model predictions computed separately",
            **metadata,
        }
    )
    bagging_selection = []
    for seed in seeds:
        options = []
        for days in settings.block_days:
            bag_oof = {members: np.full((len(train_ids), 2), np.nan) for members in settings.bag_members}
            for part, Xt, yt, Xv, indices in inner_data:
                sample_ids = part["train_event_ids"]
                scope = fold["fold_id"] + "/bag-inner-" + str(part["inner_fold"])
                member_predictions = []
                for member in range(max(settings.bag_members)):
                    draw_seed = int(digest([seed, scope, days, member])[:8], 16)
                    sampled, draw = block_sample(sample_ids, [book.decision(e) for e in sample_ids], days, draw_seed)
                    model = budget.fit_base(
                        "DeltaRidge",
                        base_seed,
                        Xt[sampled],
                        yt[sampled],
                        scope,
                        {"bootstrap": draw, "ensemble_seed": seed},
                    )
                    member_predictions.append(model.predict(Xv))
                    if member + 1 in bag_oof:
                        bag_oof[member + 1][indices] = np.mean(member_predictions, axis=0)
            scores = []
            for members in settings.bag_members:
                errors = np.abs(bag_oof[members][oof_positions] - oof_values).mean(axis=0)
                scores.append(
                    {
                        "block_days": days,
                        "members": members,
                        "score": float(np.mean(errors / scale)),
                        "cu_mae": float(errors[0]),
                        "as_mae": float(errors[1]),
                    }
                )
            chosen = select_with_tolerance(scores, settings.selection_tolerance, lambda r: (r["members"], r["score"]))
            members = []
            for member in range(chosen["members"]):
                scope = fold["fold_id"] + "/bag-full"
                draw_seed = int(digest([seed, scope, days, member])[:8], 16)
                sampled, draw = block_sample(train_ids, [book.decision(e) for e in train_ids], days, draw_seed)
                members.append(
                    budget.fit_base(
                        "DeltaRidge",
                        base_seed,
                        X[sampled],
                        Y[sampled],
                        scope,
                        {"bootstrap": draw, "ensemble_seed": seed},
                    )
                )
            name = f"Bagging_d{days}"
            values, artifact = save_predictor(
                root,
                name + "_" + str(seed),
                BaggedPredictor(members),
                X_valid,
                budget,
                {**metadata, "method": name, "selected_members": chosen["members"], "block_days": days},
                seed,
            )
            predictions[(name, seed)] = values
            artifacts.append(artifact)
            options.append(chosen)
            bagging_selection.append({"seed": seed, "block_days": days, "candidates": scores, "selected": chosen})
        choice = select_with_tolerance(
            options, settings.selection_tolerance, lambda r: (r["members"], -r["block_days"], r["score"])
        )
        predictions[("BlockBagging", seed)] = predictions[(f"Bagging_d{choice['block_days']}", seed)]
        source = next(a for a in artifacts if a["name"] == f"Bagging_d{choice['block_days']}_{seed}")
        artifacts.append(
            {
                **source,
                "name": "BlockBagging_" + str(seed),
                "method": "BlockBagging",
                "alias_of": source["name"],
                "selected_block_days": choice["block_days"],
            }
        )
    write_json(
        root / "bagging_selection.json", {"seed_results": bagging_selection, "selection_uses": "inner_forward_OOF_only"}
    )
    rows = []
    recorded_at = utc_now()
    for (method, seed), values in predictions.items():
        for event, value in zip(valid_ids, values, strict=True):
            rows.append(
                {
                    "event_id": event,
                    "fold_id": fold["fold_id"],
                    "method_id": method,
                    "seed": seed,
                    "decision_at": book.decision(event).isoformat(),
                    "fit_cutoff_at": cutoff.isoformat(),
                    "computed_at": recorded_at,
                    "cu": float(value[0]),
                    "as": float(value[1]),
                    "status": "completed",
                }
            )
    pd.DataFrame(rows).to_csv(root / "predictions.csv", index=False)
    with (root / "weight_decisions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for evidence in weight_decisions:
            stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
    result = {
        "fold_id": fold["fold_id"],
        "train_events": len(train_ids),
        "validation_events": len(valid_ids),
        "inner_oof_events": len(oof_ids),
        "warmup_events": len(train_ids) - len(oof_ids),
        "artifacts": artifacts,
        "predictions_sha256": file_hash(root / "predictions.csv"),
        "decisions_sha256": file_hash(root / "weight_decisions.jsonl"),
        "inner_oof_sha256": file_hash(root / "inner_oof.csv"),
        "inner_plan_sha256": file_hash(root / "inner_plan.json"),
        "selection_sha256": file_hash(root / "selection.json"),
        "bagging_selection_sha256": file_hash(root / "bagging_selection.json"),
    }
    write_json(root / "fold_manifest.json", result)
    return result
