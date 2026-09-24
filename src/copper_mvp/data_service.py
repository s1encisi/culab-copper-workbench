"""Versioned G1 input snapshots and local historical evidence queries."""

from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from pathlib import Path

import yaml

from copper_mas.data.leakage import find_forbidden_paths
from copper_mvp.common import WorkbenchError, digest, file_hash
from copper_mvp.data import MODE_NAMES, OFFSETS, SUFFIXES, finite
from copper_mvp.data_contracts import FeatureValue, ReplayQuery, TaskSpec, envelope, source_time
from copper_mvp.labels import LabelLedger
from copper_mvp.modeling import MODEL_VERSION


class DataService:
    def __init__(self, data, models=None):
        self.data = data
        self.models = models
        self.sources = data.paths.sources()
        path = self.sources["timing_contract"]
        if not path.is_file():
            raise WorkbenchError("缺少本地时间合同", "LOCAL_CONTRACT_MISSING")
        self.timing_hash = file_hash(path)
        self.timing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (
            self.timing.get("timezone") != "Asia/Shanghai"
            or self.timing.get("time_semantics", {}).get("available_at") != "recorded_at"
            or self.timing.get("time_semantics", {}).get("sample_at_assumed") != "recorded_at_minus_2h"
            or self.timing.get("task", {}).get("decision_at") != "origin.recorded_at"
        ):
            raise WorkbenchError("G1 尚未适配该时间合同", "TIMING_CONTRACT_UNSUPPORTED")
        self._verify_sources()

    def _verify_sources(self):
        expected = {**self.data.source_hashes, "timing_contract": self.timing_hash}
        for name, signature in expected.items():
            if not self.sources[name].is_file() or file_hash(self.sources[name]) != signature:
                raise WorkbenchError("本地数据来源已改变，请重新加载: " + name, "SOURCE_CHANGED")

    def dependencies(self, *, include_paths=False):
        items = []
        for name, path in self.sources.items():
            exists = path.is_file()
            item = {
                "source_id": name,
                "file_name": path.name,
                "exists": exists,
                "scope": "development_2024_2025",
                "access": "local_only",
                "role": "isolated_evaluation" if name in ("labels", "pairing", "index") else "input_or_contract",
                "sha256": file_hash(path) if exists else None,
                "size_bytes": path.stat().st_size if exists else None,
                "missing_reason": None if exists else "local_resource_not_configured_or_missing",
            }
            if include_paths:
                item["local_path"] = str(path)
            items.append(item)
        return envelope(
            "dependencies.g1",
            {
                "resources": items,
                "ready": all(i["exists"] for i in items),
                "legacy_dataset_version": self.data.dataset_version,
            },
            "local_resource_manifest",
        )

    def task_spec(self, event_id: str) -> TaskSpec:
        row = self.data.row(event_id)
        bad = find_forbidden_paths(dict.fromkeys(self.data.feature_columns))
        if bad:
            raise WorkbenchError("预测特征含禁止的未来字段: " + ", ".join(bad), "FUTURE_FEATURE_FIELDS")
        decision = source_time(row.decision_at)
        return TaskSpec(
            event_id=event_id,
            decision_at=decision,
            feature_cutoff_at=decision,
            input_columns=tuple(self.data.feature_columns),
            dataset_version=self.data.dataset_version,
            feature_spec_version=digest(
                {"columns": self.data.feature_columns, "contract": self.data.source_hashes["feature_contract"]}
            ),
            timing_contract_id=self.timing["contract_id"],
            timing_contract_sha256=self.timing_hash,
            assumed_sample_time=decision - timedelta(hours=2),
        )

    def snapshot(self, event_id: str, *, verify_sources=True):
        if verify_sources:
            self._verify_sources()
        spec = self.task_spec(event_id)
        row = self.data.row(event_id)
        admission = self.data.admissions.get(event_id, {})
        try:
            if (
                source_time(admission["decision_at"]) != spec.decision_at
                or source_time(admission["feature_cutoff_at"]) > spec.decision_at
                or admission["contract_id"] != spec.timing_contract_id
            ):
                raise ValueError("admission scope mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkbenchError("准入卡与事件时间/合同不一致", "AS_OF_ADMISSION") from exc
        if admission.get("admission_status") == "REJECTED":
            raise WorkbenchError("该事件未通过输入准入", "INPUT_REJECTED")
        availability = {}
        current_availability = {}
        for signal in self.data.signals:
            times = []
            tag = signal["tag"]
            for offset in OFFSETS:
                column = f"{tag}__t_minus_{offset}h"
                if finite(row[column]) is None:
                    continue
                age = finite(row.get(column + "__age_h"))
                if age is None or age < 0:
                    raise WorkbenchError("特征来源时间缺失或越过锚点: " + column, "FEATURE_AVAILABILITY")
                stamp = spec.decision_at - timedelta(hours=offset + age)
                times.append(stamp)
                if offset == 0:
                    current_availability[tag] = stamp
            availability[tag] = max(times) if times else spec.decision_at
        units = {"origin_cu_g_l": "g/L", "origin_as_mg_l": "mg/L"}
        for signal in self.data.signals:
            tag, unit = signal["tag"], signal["unit"]
            units.update(
                {
                    tag + SUFFIXES[0]: unit,
                    tag + SUFFIXES[1]: unit,
                    tag + SUFFIXES[2]: unit + "/h",
                    tag + SUFFIXES[3]: "count",
                }
            )
        units.update(dict.fromkeys(MODE_NAMES, "category"))
        features = {}
        for column in self.data.feature_columns:
            value = finite(self.data.X.loc[event_id, column])
            tag = column.split("__", 1)[0]
            # Aggregates/modes are assembled at the decision cutoff from admitted anchors.
            available = availability.get(tag, spec.decision_at)
            if column.endswith(SUFFIXES[0]):
                available = current_availability.get(tag, spec.decision_at)
            if value is None or column.endswith(SUFFIXES[3]):
                available = spec.decision_at
            if available > spec.decision_at:
                raise WorkbenchError("特征可用时间晚于决策截止: " + column, "FEATURE_AVAILABILITY")
            item = FeatureValue(
                value=value,
                unit=units[column],
                available_at=available,
                missing_reason="missing_source_observation" if value is None else None,
            )
            features[column] = item.model_dump(mode="json")
        return envelope(
            "data-snapshot.g1",
            {
                "task": spec.model_dump(mode="json"),
                "as_of": spec.decision_at.isoformat(),
                "features": features,
                "event_ids_hash": digest([event_id]),
                "input_source_hashes": {
                    k: v
                    for k, v in self.data.source_hashes.items()
                    if k in ("features", "admissions", "feature_contract")
                },
                "quality": {
                    "admission_status": admission.get("admission_status"),
                    "missing_features": sum(v["value"] is None for v in features.values()),
                },
                "lineage": {
                    "row_key": event_id,
                    "source_id": "features",
                    "derived_modes": "frozen_deterministic_mode_rules",
                    "availability_basis": "anchor_time_minus_recorded_age",
                    "process_values_controllable": False,
                },
            },
            "development_input",
        )

    @cached_property
    def labels(self):
        return LabelLedger.from_repository(self.data, self.timing["contract_id"])

    def evidence(self, event_id: str, query: ReplayQuery, *, verify_sources=True):
        snapshot = self.snapshot(event_id, verify_sources=verify_sources)
        spec = self.task_spec(event_id)
        cutoff = source_time(query.as_of) if query.as_of is not None else spec.decision_at
        if cutoff < spec.decision_at:
            raise WorkbenchError("评价截止早于该事件决策时点", "EVENT_NOT_YET_AVAILABLE")
        if self.models is None:
            raise WorkbenchError("证据查询未配置数值预测器", "MODEL_NOT_READY")
        # No outcome service is touched until the explicit-profile prediction is computed.
        prediction = self.models.predict(event_id, query.model_profile, "oof_replay", query.bundle_id)
        if prediction["fit_cutoff_at"] and source_time(prediction["fit_cutoff_at"]) > spec.decision_at:
            raise WorkbenchError("历史回放模型使用了未来训练信息", "FUTURE_MODEL")
        binding = {
            "model_version": MODEL_VERSION
            if query.model_profile != "Persistence"
            else "persistence-current-result.mvp-v1",
            "profile": query.model_profile,
            "scope": prediction["model_scope"],
            "bundle_id": prediction["bundle_id"],
            "fold_id": prediction["fold_id"],
            "fit_cutoff_at": (
                source_time(prediction["fit_cutoff_at"]).isoformat() if prediction["fit_cutoff_at"] else None
            ),
        }
        if query.model_profile != "Persistence":
            manifest = self.models.manifest(query.bundle_id)
            binding["artifact_hashes"] = {
                target: manifest["artifacts"][f"{prediction['fold_id']}:{query.model_profile}:{target}"]["sha256"]
                for target in ("cu", "as")
            }
        else:
            binding["implementation_sha256"] = file_hash(Path(__file__).with_name("modeling.py"))
        prediction_record = envelope(
            "prediction-record.g1",
            {
                "evaluation_mode": "historical_replay",
                "virtual_prediction_at": spec.decision_at.isoformat(),
                "snapshot_id": snapshot["id"],
                "event_id": event_id,
                "model": binding,
                "predictions": prediction["predictions"],
                "warnings": prediction["warnings"],
                "online_prediction_claim": False,
            },
            "historical_replay",
        )
        records = self.labels.event_records(event_id, cutoff)
        ready = len(records) == 2 and all(r.quality_eligible and r.value is not None for r in records)
        errors = {}
        if ready:
            for record in records:
                value = prediction["predictions"][record.target]["value"]
                if value is not None:
                    errors[record.target] = {
                        "unit": record.unit,
                        "signed_error": value - record.value,
                        "absolute_error": abs(value - record.value),
                    }
        status = (
            "EVALUABLE"
            if ready and len(errors) == 2
            else "INCOMPLETE_PREDICTION"
            if ready
            else "INELIGIBLE_LABELS"
            if records
            else "LABELS_NOT_AVAILABLE"
        )
        timeline = [
            {"kind": "assumed_sampling", "at": spec.assumed_sample_time.isoformat(), "is_assumption": True},
            {"kind": "decision_and_feature_cutoff", "at": spec.decision_at.isoformat(), "snapshot_id": snapshot["id"]},
            {
                "kind": "virtual_prediction",
                "at": spec.decision_at.isoformat(),
                "prediction_id": prediction_record["id"],
            },
        ]
        for record in records:
            timeline.append(
                {
                    "kind": "label_available",
                    "at": record.available_at.isoformat(),
                    "target": record.target,
                    "revision": record.revision,
                    "label_id": record.record_id,
                }
            )
        return envelope(
            "event-evidence.g1",
            {
                "evaluation_mode": "historical_replay",
                "as_of": cutoff.isoformat(),
                "event_id": event_id,
                "snapshot": snapshot,
                "prediction": prediction_record,
                "labels": [r.as_dict() for r in records],
                "evaluation": {"status": status, "errors": errors},
                "timeline": timeline,
            },
            "historical_replay",
        )
