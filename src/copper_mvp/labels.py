"""Read-only development label ledger with point-in-time revision selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta
from types import MappingProxyType

import pandas as pd

from copper_mvp.common import WorkbenchError, digest, file_hash
from copper_mvp.data import finite
from copper_mvp.data_contracts import SOURCE_TIMEZONE, UNITS, LabelRecord, envelope, source_time


class LabelLedger:
    def __init__(self, records: Iterable[LabelRecord]):
        ordered = tuple(sorted(records, key=lambda r: (r.event_id, r.target, r.revision)))
        groups = defaultdict(list)
        for record in ordered:
            chain = groups[(record.event_id, record.target)]
            if not chain:
                if record.revision != 1:
                    raise WorkbenchError("标签修订缺少初始版本", "LABEL_REVISION")
            else:
                previous = chain[-1]
                if (
                    record.revision != previous.revision + 1
                    or record.supersedes != previous.record_id
                    or record.available_at < previous.available_at
                    or record.pair_id != previous.pair_id
                    or record.decision_at != previous.decision_at
                ):
                    raise WorkbenchError("标签修订链冲突", "LABEL_REVISION")
            chain.append(record)
        self.records = ordered
        self._groups = MappingProxyType({key: tuple(chain) for key, chain in groups.items()})
        self.version = digest([r.record_id for r in ordered])

    def with_revision(self, revision: LabelRecord) -> LabelLedger:
        """Return a new ledger; previously issued snapshots and records remain valid."""
        return LabelLedger((*self.records, revision))

    @classmethod
    def from_repository(cls, data, contract_id: str) -> LabelLedger:
        sources = data.paths.sources()
        names = ("labels", "pairing", "index")
        if any(not sources[name].is_file() for name in names):
            raise WorkbenchError("开发标签账本或配对索引不可用，请配置本地依赖", "LOCAL_LABELS_MISSING")
        hashes = {name: file_hash(sources[name]) for name in names}
        if hashes["index"] != data.source_hashes["index"]:
            raise WorkbenchError("资格索引已改变，请重新加载", "SOURCE_CHANGED")
        pairs = pd.read_csv(sources["pairing"])
        if "target_available_at" not in pairs:
            raise WorkbenchError("配对索引缺少标签可得时间", "LABEL_SCHEMA")
        try:
            times = [source_time(t) for t in pairs.target_available_at.dropna()]
        except (TypeError, ValueError) as exc:
            raise WorkbenchError("标签可得时间格式错误", "LABEL_CONTRACT") from exc
        if any(t.astimezone(SOURCE_TIMEZONE).year not in (2024, 2025) for t in times):
            raise WorkbenchError("标签来源越过开发数据年份边界", "DATA_PERIOD")
        outcomes = pd.read_csv(sources["labels"])
        needed_outcomes = {"pair_id", "contract_id", "target_cu_g_l", "target_as_mg_l"}
        needed_pairs = {
            "pair_id",
            "contract_id",
            "origin_event_id",
            "origin_recorded_at",
            "target_available_at",
            "target_sample_at_assumed",
        }
        if not needed_outcomes <= set(outcomes) or not needed_pairs <= set(pairs):
            raise WorkbenchError("开发标签来源字段不完整", "LABEL_SCHEMA")
        if (
            outcomes.pair_id.duplicated().any()
            or pairs.pair_id.duplicated().any()
            or pairs.origin_event_id.duplicated().any()
            or not set(outcomes.pair_id) <= set(pairs.pair_id)
            or set(pairs.origin_event_id) != set(data.frame.index)
            or not outcomes.contract_id.eq(contract_id).all()
            or not pairs.contract_id.eq(contract_id).all()
        ):
            raise WorkbenchError("开发标签身份、合同或连接关系不一致", "LABEL_JOIN")
        joined = pairs.merge(outcomes, on="pair_id", how="left", validate="one_to_one", suffixes=("", "_outcome"))
        source_version = digest(hashes)
        records = []
        outcome_ids = set(outcomes.pair_id)
        for row in joined.itertuples(index=False):
            present = row.pair_id in outcome_ids
            if not present:
                continue
            try:
                decision = source_time(row.origin_recorded_at)
                available = source_time(row.target_available_at)
                assumed = source_time(row.target_sample_at_assumed)
                if decision != source_time(data.row(row.origin_event_id).decision_at):
                    raise ValueError("origin time mismatch")
                if decision.astimezone(SOURCE_TIMEZONE).year not in (2024, 2025):
                    raise ValueError("outside development scope")
                for target, column in (("cu", "target_cu_g_l"), ("as", "target_as_mg_l")):
                    value = finite(getattr(row, column))
                    missing = None
                    if value is None or value < 0:
                        value = None
                        missing = "missing_or_invalid_source_value"
                    records.append(
                        LabelRecord(
                            event_id=row.origin_event_id,
                            pair_id=row.pair_id,
                            target=target,
                            value=value,
                            unit=UNITS[target],
                            decision_at=decision,
                            available_at=available,
                            assumed_sample_time=assumed,
                            source_version=source_version,
                            quality_eligible=row.origin_event_id in data.primary_ids and value is not None,
                            missing_reason=missing,
                        )
                    )
            except (ValueError, TypeError) as exc:
                raise WorkbenchError("标签时间、单位或数值合同不成立", "LABEL_CONTRACT") from exc
        if any(file_hash(sources[name]) != value for name, value in hashes.items()):
            raise WorkbenchError("加载期间标签来源发生变化，请重新加载", "SOURCE_CHANGED")
        result = cls(records)
        result._sources = {name: (sources[name], value) for name, value in hashes.items()}
        return result

    def verify_sources(self):
        for name, (path, expected) in getattr(self, "_sources", {}).items():
            if not path.is_file() or file_hash(path) != expected:
                raise WorkbenchError("标签来源已改变，请重新加载: " + name, "SOURCE_CHANGED")

    def latest(self, as_of) -> dict[tuple[str, str], LabelRecord]:
        self.verify_sources()
        cutoff = source_time(as_of)
        result = {}
        for key, chain in self._groups.items():
            available = [r for r in chain if r.available_at <= cutoff]
            if available:
                result[key] = available[-1]
        return result

    def event_records(self, event_id: str, as_of) -> tuple[LabelRecord, ...]:
        latest = self.latest(as_of)
        return tuple(latest[(event_id, target)] for target in ("cu", "as") if (event_id, target) in latest)

    def eligible_events(self, as_of, days: int) -> dict[str, tuple[LabelRecord, ...]]:
        cutoff = source_time(as_of)
        start = cutoff - timedelta(days=days)
        grouped = defaultdict(list)
        for record in self.latest(cutoff).values():
            if start < record.decision_at <= cutoff and record.value is not None and record.quality_eligible:
                grouped[record.event_id].append(record)
        return {
            key: tuple(sorted(value, key=lambda r: r.target))
            for key, value in grouped.items()
            if {r.target for r in value} == {"cu", "as"}
        }

    def snapshot(self, as_of, *, days=30, limit=60, minimum=60, coverage=None) -> dict:
        eligible = self.eligible_events(as_of, days)
        keys = set(eligible)
        coverage_counts = {}
        if coverage is not None:
            for model, events in coverage.items():
                coverage_counts[model] = {"available": len(set(events) & keys), "missing": len(keys - set(events))}
            keys = set.intersection(keys, *(set(events) for events in coverage.values())) if coverage else set()
        selected = sorted(keys, key=lambda e: (eligible[e][0].decision_at, e), reverse=True)[:limit]
        rows = [record.as_dict() for event in selected for record in eligible[event]]
        return envelope(
            "label-snapshot.g1",
            {
                "evaluation_mode": "historical_replay",
                "as_of": source_time(as_of).isoformat(),
                "window_start": (source_time(as_of) - timedelta(days=days)).isoformat(),
                "window_basis": "event_decision_at",
                "ledger_version": digest([row["id"] for row in rows]),
                "status": "READY_FOR_OFFLINE_EVALUATION" if len(selected) >= minimum else "INSUFFICIENT_LABELS",
                "minimum_events": minimum,
                "window_mature_events": len(eligible),
                "selected_events": len(selected),
                "event_ids": selected,
                "common_event_hash": digest(sorted(selected)),
                "coverage": coverage_counts,
                "labels": rows,
                "routing_action": "NONE",
            },
            "development_ledger",
        )
