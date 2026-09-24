from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from copper_mas.agents.mode import infer_process_mode
from copper_mvp.common import LocalDataPaths, WorkbenchError, digest, file_hash, safe

SUFFIXES = ("__t_minus_0h", "__mean_12h", "__slope_per_h_12h", "__valid_count_12h")
TARGETS = ("cu", "as")
MODE_NAMES = ("mode_p1_stage12", "mode_p2_stage12", "mode_s3", "mode_s4")
MODE_CODES = {"OFF": 0.0, "ON": 1.0, "UNKNOWN": 2.0}
OFFSETS = (0, 2, 4, 6, 8, 10, 12)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(("true", "1"))


class DataRepository:
    def __init__(
        self,
        data_dir: Path | None = None,
        evidence_dir: Path | None = None,
        *,
        contract_dir: Path | None = None,
        label_dir: Path | None = None,
    ):
        self.paths = LocalDataPaths.resolve(data_dir, evidence_dir, contract_dir, label_dir)
        self.data_dir = self.paths.data_dir
        self.evidence_dir = self.paths.evidence_dir
        self.label_dir = self.paths.label_dir
        sources = self.paths.sources()
        required = ("features", "admissions", "index", "folds", "feature_contract")
        missing = [name for name in required if not sources[name].is_file()]
        if missing:
            raise WorkbenchError("缺少本地数据依赖: " + ", ".join(missing), "LOCAL_DATA_MISSING")
        self.source_hashes = {name: file_hash(sources[name]) for name in required}
        config_path = sources["feature_contract"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.signals = [dict(v, group=g) for g, group in config["groups"].items() for v in group["features"].values()]
        self.tags = [s["tag"] for s in self.signals]
        self.numeric_columns = ["origin_cu_g_l", "origin_as_mg_l"] + [t + s for t in self.tags for s in SUFFIXES]
        self.feature_columns = self.numeric_columns + list(MODE_NAMES)
        frame = pd.read_csv(self.data_dir / "core_feature_matrix_v2.csv", low_memory=False)
        if frame.origin_event_id.duplicated().any() or set(self.numeric_columns) - set(frame.columns):
            raise WorkbenchError("事件或特征定义不一致", "DATA_SCHEMA")
        frame["decision_at"] = pd.to_datetime(frame.decision_at)
        if not frame.decision_at.dt.year.isin((2024, 2025)).all():
            raise WorkbenchError("本工作台使用 2024—2025 开发数据", "DATA_PERIOD")
        self.frame = frame.sort_values("decision_at", kind="stable").set_index("origin_event_id", drop=False)
        self.index = pd.read_csv(self.data_dir / "training_evaluation_index_v2.csv")
        admission_frame = pd.read_csv(self.data_dir / "as_of_admission_card_core_v2.csv")
        self.admissions = admission_frame.set_index("origin_event_id").to_dict("index")
        self.primary_index = self.index.loc[truthy(self.index.tier1_core_primary_prospective)].copy()
        self.primary_ids = set(self.primary_index.origin_event_id)
        self.cv = pd.read_csv(self.data_dir / "cv_fold_manifest_v2.csv")
        self.cv = self.cv[self.cv.origin_event_id.isin(self.primary_ids)].copy()
        validation = self.cv[self.cv.fold_role.eq("VALIDATION")]
        if validation.origin_event_id.duplicated().any():
            raise WorkbenchError("OOF 事件身份重复", "DATA_FOLDS")
        self.fold_for = dict(zip(validation.origin_event_id, validation.fold_id, strict=True))
        self.dataset_version = digest(
            {
                "features": file_hash(self.data_dir / "core_feature_matrix_v2.csv"),
                "folds": file_hash(self.data_dir / "cv_fold_manifest_v2.csv"),
                "contract": file_hash(config_path),
            }
        )
        mode_prefixes = (
            "phase1_stage12_",
            "phase2_stage12_",
            "stage3_current_",
            "stage3_voltage_",
            "stage3_flow_",
            "stage4_current_",
            "stage4_voltage_",
            "stage4_flow_",
        )
        mode_cols = [
            c
            for c in self.frame.columns
            if c.startswith(mode_prefixes) and "__t_minus_" in c and not c.endswith("__age_h")
        ]
        self.mode_cards: dict[str, dict] = {}
        categories = []
        records = self.frame[mode_cols].to_dict("records")
        for (event_id, decision), row in zip(
            self.frame[["origin_event_id", "decision_at"]].itertuples(index=False, name=None), records, strict=True
        ):
            card = infer_process_mode(
                origin_event_id=event_id, decision_at=decision.to_pydatetime(), feature_row=row
            ).model_dump(mode="json")
            card["states"] = [part.rsplit("_", 1)[-1] for part in card["mode_code"].split("__")]
            self.mode_cards[event_id] = card
            categories.append([MODE_CODES[s] for s in card["states"]])
        numeric = (
            self.frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        )
        self.X = pd.DataFrame(
            np.column_stack((numeric.to_numpy(float), np.array(categories))),
            index=self.frame.index,
            columns=self.feature_columns,
        )
        self.summaries = [self._summary(event_id) for event_id in self.frame.index]
        self._targets = None
        if any(file_hash(sources[name]) != value for name, value in self.source_hashes.items()):
            raise WorkbenchError("装配期间数据来源发生变化，请重新加载", "SOURCE_CHANGED")

    def row(self, event_id: str) -> pd.Series:
        if event_id not in self.frame.index:
            raise WorkbenchError("没有找到该历史事件", "EVENT_NOT_FOUND")
        return self.frame.loc[event_id]

    def _summary(self, event_id: str) -> dict:
        row = self.row(event_id)
        supported = [
            s
            for s in (3, 4)
            if all(
                (v := finite(row[f"stage{s}_{tag}__t_minus_0h"])) is not None and v > 0
                for tag in ("current_a", "voltage_v")
            )
        ]
        mode = self.mode_cards[event_id]
        admission = self.admissions.get(event_id, {})
        warnings = list(mode["warnings"])
        for field in ("reason_codes", "source_quality_warnings"):
            raw = admission.get(field)
            if isinstance(raw, str):
                warnings.extend(json.loads(raw))
        if not bool(row.origin_result_quality_eligible):
            warnings.append("CURRENT_RESULT_QUALITY_WARNING")
        return safe(
            {
                "event_id": event_id,
                "decision_at": row.decision_at.isoformat(),
                "year": row.decision_at.year,
                "cu": row.origin_cu_g_l,
                "as": row.origin_as_mg_l,
                "mode": mode["states"],
                "mode_code": mode["mode_code"],
                "warnings": sorted(set(warnings)),
                "admission_status": admission.get("admission_status", "WARNING"),
                "fold_id": self.fold_for.get(event_id),
                "primary": event_id in self.primary_ids,
                "supported_stages": supported,
                "optimization_ready": event_id in self.fold_for and bool(supported),
            }
        )

    def events(
        self, year: int | None = None, query: str = "", scope: str = "all", offset: int = 0, limit: int = 40
    ) -> dict:
        items = list(reversed(self.summaries))
        if year:
            items = [r for r in items if r["year"] == year]
        if query:
            query = query.lower().strip()
            items = [r for r in items if query in r["event_id"].lower() or query in r["decision_at"]]
        if scope == "oof":
            items = [r for r in items if r["fold_id"]]
        if scope == "dual":
            items = [r for r in items if r["fold_id"] and len(r["supported_stages"]) == 2]
        return {"total": len(items), "offset": offset, "items": items[offset : offset + limit]}

    def context(self, event_id: str) -> dict:
        row = self.row(event_id)
        signals = []
        for signal in self.signals:
            tag = signal["tag"]
            signals.append(
                {
                    **signal,
                    "current": finite(row[tag + SUFFIXES[0]]),
                    "mean": finite(row[tag + SUFFIXES[1]]),
                    "slope": finite(row[tag + SUFFIXES[2]]),
                    "count": int(row[tag + SUFFIXES[3]]),
                    "anchors": [{"hours": -k, "value": finite(row[f"{tag}__t_minus_{k}h"])} for k in reversed(OFFSETS)],
                }
            )
        history = self.frame[self.frame.decision_at.le(row.decision_at)].tail(60)
        return safe(
            {
                **self._summary(event_id),
                "signals": signals,
                "mode_evidence": self.mode_cards[event_id],
                "history": [
                    {"time": r.decision_at.isoformat(), "cu": r.origin_cu_g_l, "as": r.origin_as_mg_l}
                    for r in history.itertuples()
                ],
                "dataset_version": self.dataset_version,
                "feature_count": 114,
            }
        )

    def overview(self) -> dict:
        selected = [r for r in self.summaries if r["primary"]]
        return {
            "events": len(self.frame),
            "primary": len(selected),
            "oof": len(self.fold_for),
            "numeric_features": 110,
            "mode_features": 4,
            "signals": 27,
            "dual_stage_cases": sum(r["fold_id"] is not None and len(r["supported_stages"]) == 2 for r in selected),
            "start": self.frame.decision_at.min().isoformat(),
            "end": self.frame.decision_at.max().isoformat(),
            "dataset_version": self.dataset_version,
            "latest": self.summaries[-1],
            "signal_catalog": self.signals,
        }

    def training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._targets is None:
            folder = self.label_dir
            pair_times = pd.read_csv(
                folder / "event_pair_index_v2.csv", usecols=["pair_id", "target_available_at"]
            ).set_index("pair_id")
            if pair_times.index.duplicated().any():
                raise WorkbenchError("标签配对身份重复", "LABEL_JOIN")
            available_times = pd.to_datetime(pair_times.target_available_at, errors="raise")
            if not available_times.dropna().dt.year.isin((2024, 2025)).all():
                raise WorkbenchError("标签来源越过开发数据年份边界", "DATA_PERIOD")
            primary_available = available_times.loc[self.primary_index.pair_id]
            if primary_available.isna().any() or primary_available.max() > self.frame.decision_at.max():
                raise WorkbenchError("全开发期标签缺少可得时间或晚于拟合截止", "TEMPORAL_SPLIT")
            outcomes = pd.read_csv(
                folder / "outcome_ledger_v2.csv", usecols=["pair_id", "target_cu_g_l", "target_as_mg_l"]
            )
            joined = self.primary_index.merge(outcomes, on="pair_id", validate="one_to_one").set_index(
                "origin_event_id"
            )
            if len(joined) != len(self.primary_ids):
                raise WorkbenchError("开发标签连接不完整", "LABEL_JOIN")
            targets = joined[["target_cu_g_l", "target_as_mg_l"]].astype(float)
            targets.columns = list(TARGETS)
            if not np.isfinite(targets.to_numpy()).all():
                raise WorkbenchError("开发标签含无效数值", "LABEL_VALUES")
            for fold_id in sorted(self.cv.fold_id.unique()):
                rows = self.cv[self.cv.fold_id.eq(fold_id)]
                train = rows[rows.fold_role.eq("TRAIN")]
                valid = rows[rows.fold_role.eq("VALIDATION")]
                cutoff = pd.Timestamp(valid.fold_fit_cutoff_at.iloc[0])
                train_available = pd.to_datetime(pair_times.loc[train.pair_id, "target_available_at"])
                if (
                    train_available.isna().any()
                    or pd.to_datetime(train.decision_at).max() >= cutoff
                    or train_available.max() > cutoff
                ):
                    raise WorkbenchError("训练时间或标签可用时间越过截止点", "TEMPORAL_SPLIT")
            self._targets = targets
        return self.X.loc[self._targets.index], self._targets.copy()

    def training_ids(self, fold_id: str | None) -> list[str]:
        if fold_id is None:
            return [event for event in self.frame.index if event in self.primary_ids]
        return self.cv.loc[self.cv.fold_id.eq(fold_id) & self.cv.fold_role.eq("TRAIN"), "origin_event_id"].tolist()

    def candidate_matrix(self, event_id: str, stages: list[int], candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=float)
        matrix = np.repeat(self.X.loc[[event_id]].to_numpy(float), len(candidates), axis=0)
        row = self.row(event_id)
        for j, stage in enumerate(stages):
            tag = f"stage{stage}_current_a"
            values = np.array([row[f"{tag}__t_minus_{k}h"] for k in OFFSETS], dtype=float)
            mask = np.isfinite(values)
            if not mask[0]:
                raise WorkbenchError("该段缺少当前电流", "CURRENT_MISSING")
            altered = np.repeat(values[None, :], len(candidates), axis=0)
            altered[:, 0] = candidates[:, j]
            matrix[:, self.feature_columns.index(tag + SUFFIXES[0])] = candidates[:, j]
            matrix[:, self.feature_columns.index(tag + SUFFIXES[1])] = np.mean(altered[:, mask], axis=1)
            if mask.sum() >= 2:
                times = -np.array(OFFSETS, dtype=float)[mask]
                centered = times - times.mean()
                matrix[:, self.feature_columns.index(tag + SUFFIXES[2])] = (
                    altered[:, mask] @ centered / (centered @ centered)
                )
        return matrix
