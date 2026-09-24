"""Time boundaries and label access shared by nested fitting and delayed feedback."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from copper_mvp.common import WorkbenchError, digest
from copper_mvp.data_contracts import source_time


class EnsembleData:
    def __init__(self, data, ledger):
        self.data = data
        self.ledger = ledger
        self.decisions = {e: source_time(data.row(e).decision_at) for e in data.X.index}
        self.modes = {e: data.mode_cards[e]["mode_code"] for e in data.X.index}
        self.chains = defaultdict(list)
        for record in ledger.records:
            self.chains[(record.event_id, record.target)].append(record)

    def decision(self, event):
        return self.decisions[event]

    def mode(self, event):
        return self.modes[event]

    def record(self, event, target, cutoff):
        cutoff = source_time(cutoff) if isinstance(cutoff, str) else cutoff
        value = next((r for r in reversed(self.chains[(event, target)]) if r.available_at <= cutoff), None)
        return value if value and value.quality_eligible and value.value is not None else None

    def mature(self, events, cutoff):
        cutoff = source_time(cutoff)
        return [e for e in events if self.decision(e) < cutoff and all(self.record(e, t, cutoff) for t in ("cu", "as"))]

    def y(self, events, cutoff):
        records = [[self.record(e, t, cutoff) for t in ("cu", "as")] for e in events]
        if any(r is None for pair in records for r in pair):
            raise WorkbenchError("训练标签在该截止时间不可得", "ENSEMBLE_IMMATURE_LABEL")
        return np.array([[r.value for r in pair] for pair in records])

    def label_ids(self, events, cutoff):
        result = [self.record(e, t, cutoff) for e in events for t in ("cu", "as")]
        if any(r is None for r in result):
            raise WorkbenchError("训练标签版本尚不可得", "ENSEMBLE_IMMATURE_LABEL")
        return [r.record_id for r in result]

    def X(self, events):
        return self.data.X.loc[events].to_numpy(float, copy=True)

    def evidence(self, events, cutoff):
        records = [self.record(e, t, cutoff) for e in events for t in ("cu", "as")]
        if not records or any(r is None for r in records):
            raise WorkbenchError("训练证据缺少成熟标签", "ENSEMBLE_IMMATURE_LABEL")
        return {
            "fit_cutoff_at": source_time(cutoff).isoformat(),
            "train_count": len(events),
            "train_event_hash": digest(events),
            "label_record_hash": digest([r.record_id for r in records]),
            "latest_training_decision": max(self.decision(e) for e in events).isoformat(),
            "latest_label_available_at": max(r.available_at for r in records).isoformat(),
        }

    def inner_plan(self, train_events, outer_cutoff, settings):
        events = sorted(train_events, key=lambda e: (self.decision(e), e))
        if self.mature(events, outer_cutoff) != events:
            raise WorkbenchError("外层训练集合包含未成熟或未来标签", "ENSEMBLE_OUTER_SPLIT")
        if len(events) < 80:
            raise WorkbenchError("嵌套训练需要至少 80 个外层训练事件", "ENSEMBLE_INNER_SUPPORT")
        cuts = [
            self.decision(events[min(len(events) - 1, int(len(events) * f))]) for f in settings.inner_boundaries[:-1]
        ]
        plan = []
        for index, cutoff in enumerate(cuts):
            end = cuts[index + 1] if index + 1 < len(cuts) else source_time(outer_cutoff)
            valid = [e for e in events if cutoff <= self.decision(e) < end]
            train = self.mature(events, cutoff)
            if len(train) < 30 or len(valid) < 10 or set(train) & set(valid):
                raise WorkbenchError("内层时间折的训练或验证支持不足", "ENSEMBLE_INNER_SUPPORT")
            plan.append(
                {
                    "inner_fold": index + 1,
                    "train_event_ids": train,
                    "validation_event_ids": valid,
                    "validation_event_hash": digest(valid),
                    **self.evidence(train, cutoff),
                }
            )
        return plan


def block_sample(events, decisions, days, seed):
    dates = pd.to_datetime(decisions, utc=True).tz_convert("Asia/Shanghai").normalize().asi8
    bins = dates // (86400 * 10**9 * days)
    blocks = np.unique(bins)
    rng = np.random.default_rng(seed)
    draws = rng.choice(blocks, size=len(blocks), replace=True)
    indices = np.sort(np.concatenate([np.flatnonzero(bins == block) for block in draws]))
    return indices, {
        "seed": seed,
        "block_days": days,
        "source_blocks": len(blocks),
        "drawn_blocks": draws.tolist(),
        "sample_count": len(indices),
        "sample_event_hash": digest([events[i] for i in indices]),
        "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
    }


def select_with_tolerance(rows, tolerance, priority):
    best = min(row["score"] for row in rows)
    candidates = [row for row in rows if row["score"] <= best + tolerance * max(abs(best), 1e-12)]
    return min(candidates, key=priority)
