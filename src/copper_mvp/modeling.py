from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from copper_mvp.common import WorkbenchError, file_hash, safe, utc_now, write_json
from copper_mvp.data import TARGETS, DataRepository, finite

FAMILIES = ("DeltaRidge", "DeltaHGB")
MODEL_VERSION = "delta-114-v1"


def make_model(family: str):
    if family == "DeltaRidge":
        preprocessing = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    list(range(110)),
                ),
                (
                    "modes",
                    OneHotEncoder(categories=[[0.0, 1.0, 2.0]] * 4, handle_unknown="ignore", sparse_output=False),
                    list(range(110, 114)),
                ),
            ]
        )
        return Pipeline([("preprocess", preprocessing), ("regressor", Ridge(alpha=1000.0))])
    if family == "DeltaHGB":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            categorical_features=[False] * 110 + [True] * 4,
            random_state=20260905,
        )
    raise WorkbenchError("没有该模型预设", "MODEL_NOT_FOUND")


def metrics(y: np.ndarray, predicted: np.ndarray) -> dict:
    if not np.isfinite(predicted).all():
        raise WorkbenchError("模型输出出现非有限值", "NONFINITE_PREDICTION")
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
        "r2": float(r2_score(y, predicted)) if len(y) > 1 else None,
        "bias": float(np.mean(predicted - y)),
    }


class ModelManager:
    def __init__(self, root: Path, data: DataRepository):
        self.root = Path(root) / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = data
        self._cache: dict[tuple, object] = {}

    def catalog(self) -> list[dict]:
        bundles = []
        for path in self.root.glob("*/manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") == "completed":
                bundles.append(manifest)
        return sorted(bundles, key=lambda b: b["created_at"], reverse=True)

    def manifest(self, bundle_id: str | None = None) -> dict:
        bundles = self.catalog()
        bundle = (
            next((b for b in bundles if b["bundle_id"] == bundle_id), None)
            if bundle_id
            else (bundles[0] if bundles else None)
        )
        if bundle is None:
            raise WorkbenchError("请先在模型实验中训练响应模型", "MODEL_NOT_READY")
        if bundle["dataset_version"] != self.data.dataset_version or bundle["model_version"] != MODEL_VERSION:
            raise WorkbenchError("模型与当前数据或特征版本不一致，请重新训练", "MODEL_VERSION_MISMATCH")
        return bundle

    def train(self, bundle_id: str, progress: Callable[[str, dict, float], None]) -> dict:
        X, y = self.data.training_data()
        folder = self.root / bundle_id
        folder.mkdir(parents=True, exist_ok=True)
        outputs = []
        fold_metrics = []
        artifacts = {}
        for fold_id in sorted(self.data.cv.fold_id.unique()):
            rows = self.data.cv[self.data.cv.fold_id.eq(fold_id)]
            train_ids = rows.loc[rows.fold_role.eq("TRAIN"), "origin_event_id"].tolist()
            valid_ids = rows.loc[rows.fold_role.eq("VALIDATION"), "origin_event_id"].tolist()
            train_X = X.loc[train_ids].to_numpy(float)
            valid_X = X.loc[valid_ids].to_numpy(float)
            for target_index, target in enumerate(TARGETS):
                valid_y = y.loc[valid_ids, target].to_numpy(float)
                current = valid_X[:, target_index]
                fold_metrics.append(
                    {"fold_id": fold_id, "model": "Persistence", "target": target, **metrics(valid_y, current)}
                )
                for event, actual, predicted in zip(valid_ids, valid_y, current, strict=True):
                    outputs.append(
                        {
                            "event_id": event,
                            "fold_id": fold_id,
                            "target": target,
                            "model": "Persistence",
                            "actual": actual,
                            "predicted": predicted,
                        }
                    )
                delta = y.loc[train_ids, target].to_numpy(float) - train_X[:, target_index]
                for family in FAMILIES:
                    start = perf_counter()
                    model = make_model(family)
                    model.fit(train_X, delta)
                    prediction = current + model.predict(valid_X)
                    measure = metrics(valid_y, prediction)
                    filename = f"{fold_id}_{family}_{target}.joblib"
                    joblib.dump(model, folder / filename, compress=3)
                    artifacts[f"{fold_id}:{family}:{target}"] = {
                        "file": filename,
                        "sha256": file_hash(folder / filename),
                        "fit_cutoff_at": str(rows.fold_fit_cutoff_at.iloc[0]),
                        "train_count": len(train_ids),
                    }
                    fold_metrics.append({"fold_id": fold_id, "model": family, "target": target, **measure})
                    for event, actual, predicted in zip(valid_ids, valid_y, prediction, strict=True):
                        outputs.append(
                            {
                                "event_id": event,
                                "fold_id": fold_id,
                                "target": target,
                                "model": family,
                                "actual": actual,
                                "predicted": predicted,
                            }
                        )
                    progress(f"A3 · {fold_id} · {target.upper()} · {family}", measure, (perf_counter() - start) * 1000)
        oof = pd.DataFrame(outputs)
        metric_rows = []
        for (family, target), group in oof.groupby(["model", "target"]):
            metric_rows.append(
                {"model": family, "target": target, **metrics(group.actual.to_numpy(), group.predicted.to_numpy())}
            )
        for row in metric_rows:
            baseline = next(
                r["mae"] for r in metric_rows if r["model"] == "Persistence" and r["target"] == row["target"]
            )
            row["skill_mae"] = 1 - row["mae"] / baseline if baseline else 0
        defaults = {
            t: min(
                (r for r in metric_rows if r["target"] == t and r["model"] in FAMILIES),
                key=lambda r: (round(r["mae"], 6), 0 if r["model"] == "DeltaRidge" else 1),
            )["model"]
            for t in TARGETS
        }
        whole_X = X.to_numpy(float)
        for index, target in enumerate(TARGETS):
            for family in FAMILIES:
                start = perf_counter()
                model = make_model(family)
                model.fit(whole_X, y[target].to_numpy(float) - whole_X[:, index])
                filename = f"DEVELOPMENT_{family}_{target}.joblib"
                joblib.dump(model, folder / filename, compress=3)
                artifacts[f"DEVELOPMENT:{family}:{target}"] = {
                    "file": filename,
                    "sha256": file_hash(folder / filename),
                    "fit_cutoff_at": self.data.frame.decision_at.max().isoformat(),
                    "train_count": len(X),
                }
                progress(
                    f"A3 · 全开发期 · {target.upper()} · {family}",
                    {"train_count": len(X)},
                    (perf_counter() - start) * 1000,
                )
        oof["decision_at"] = oof.event_id.map(self.data.frame.decision_at).astype(str)
        oof.to_csv(folder / "oof_predictions.csv", index=False)
        manifest = safe(
            {
                "bundle_id": bundle_id,
                "status": "completed",
                "created_at": utc_now(),
                "dataset_version": self.data.dataset_version,
                "model_version": MODEL_VERSION,
                "feature_columns": self.data.feature_columns,
                "numeric_features": 110,
                "categorical_features": 4,
                "train_count": len(X),
                "oof_events": oof.event_id.nunique(),
                "default_response": defaults,
                "metrics": metric_rows,
                "fold_metrics": fold_metrics,
                "artifacts": artifacts,
                "oof_predictions_sha256": file_hash(folder / "oof_predictions.csv"),
                "model_configs": {
                    "DeltaRidge": {"alpha": 1000},
                    "DeltaHGB": {
                        "learning_rate": 0.05,
                        "max_iter": 200,
                        "max_leaf_nodes": 15,
                        "min_samples_leaf": 20,
                        "l2_regularization": 1,
                        "early_stopping": False,
                    },
                },
                "scopes": ["oof_replay", "development_analysis"],
            }
        )
        write_json(folder / "manifest.json", manifest)
        return manifest

    def resolve(self, event_id: str, profile: str, scope: str, bundle_id: str | None = None) -> dict:
        manifest = self.manifest(bundle_id)
        fold = self.data.fold_for.get(event_id) if scope == "oof_replay" else "DEVELOPMENT"
        if fold is None:
            raise WorkbenchError("该事件没有 OOF 模型，可选择 Persistence 或开发分析", "NO_OOF_MODEL")
        models = {}
        names = {}
        cutoff = None
        for target in TARGETS:
            name = manifest["default_response"][target] if profile == "auto" else profile
            artifact = manifest["artifacts"].get(f"{fold}:{name}:{target}")
            if artifact is None:
                raise WorkbenchError("该响应模型不可用", "MODEL_NOT_FOUND")
            path = self.root / manifest["bundle_id"] / artifact["file"]
            signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            if signature not in self._cache:
                if file_hash(path) != artifact["sha256"]:
                    raise WorkbenchError("模型文件校验失败", "MODEL_HASH_MISMATCH")
                self._cache[signature] = joblib.load(path)
            models[target] = self._cache[signature]
            names[target] = name
            cutoff = artifact["fit_cutoff_at"]
        return {"models": models, "names": names, "fold_id": fold, "fit_cutoff_at": cutoff, "manifest": manifest}

    @staticmethod
    def predict_matrix(resolved: dict, X: np.ndarray) -> np.ndarray:
        values = np.column_stack([X[:, i] + resolved["models"][target].predict(X) for i, target in enumerate(TARGETS)])
        if not np.isfinite(values).all():
            raise WorkbenchError("响应模型输出出现非有限值", "NONFINITE_PREDICTION")
        return values

    def predict(
        self, event_id: str, profile: str = "auto", scope: str = "oof_replay", bundle_id: str | None = None
    ) -> dict:
        context = self.data.context(event_id)
        matrix = self.data.X.loc[[event_id]].to_numpy(float)
        warnings = list(context["warnings"])
        resolved = None
        if profile != "Persistence":
            try:
                resolved = self.resolve(event_id, profile, scope, bundle_id)
            except WorkbenchError as exc:
                if profile != "auto" or exc.code not in ("MODEL_NOT_READY", "NO_OOF_MODEL"):
                    raise
                warnings.append(exc.code)
        predictions = {}
        for i, target in enumerate(TARGETS):
            current = finite(matrix[0, i])
            if current is None:
                predictions[target] = {
                    "current": None,
                    "value": None,
                    "delta": None,
                    "model": None,
                    "unit": "g/L" if target == "cu" else "mg/L",
                }
                warnings.append(f"{target.upper()}_CURRENT_MISSING")
                continue
            delta = float(resolved["models"][target].predict(matrix)[0]) if resolved else 0.0
            value = current + delta
            if not np.isfinite(value):
                raise WorkbenchError("预测出现非有限值", "NONFINITE_PREDICTION")
            if value < 0:
                warnings.append(f"{target.upper()}_NEGATIVE_MODEL_PREDICTION")
            predictions[target] = {
                "current": current,
                "value": value,
                "delta": delta,
                "model": resolved["names"][target] if resolved else "Persistence",
                "unit": "g/L" if target == "cu" else "mg/L",
            }
        return safe(
            {
                "kind": "prediction",
                "event_id": event_id,
                "decision_at": context["decision_at"],
                "predictions": predictions,
                "mode": context["mode"],
                "mode_evidence": context["mode_evidence"],
                "warnings": sorted(set(warnings)),
                "model_scope": scope if resolved else "reference",
                "fold_id": resolved["fold_id"] if resolved else context["fold_id"],
                "fit_cutoff_at": resolved["fit_cutoff_at"] if resolved else None,
                "bundle_id": resolved["manifest"]["bundle_id"] if resolved else None,
                "validation_metrics": resolved["manifest"]["metrics"] if resolved else [],
                "dataset_version": self.data.dataset_version,
            }
        )

    def series(self, bundle_id: str, target: str = "cu") -> list[dict]:
        manifest = self.manifest(bundle_id)
        path = self.root / manifest["bundle_id"] / "oof_predictions.csv"
        frame = pd.read_csv(path)
        frame = frame[frame.target.eq(target)]
        pivot = (
            frame.pivot(index=["event_id", "decision_at", "actual"], columns="model", values="predicted")
            .reset_index()
            .sort_values("decision_at")
        )
        step = max(1, len(pivot) // 180)
        return safe(pivot.iloc[::step].to_dict("records"))
