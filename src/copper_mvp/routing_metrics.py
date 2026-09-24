"""Point-in-time, common-cohort metrics for historical predictor-policy replay."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from copper_mvp.common import WorkbenchError, digest, safe
from copper_mvp.data_contracts import source_time
from copper_mvp.model_evaluation import numerical_metrics


class RoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = "routing.fixed.g5a.v2"
    short_events: int = Field(default=60, ge=1)
    short_days: int = Field(default=30, ge=1)
    long_events: int = Field(default=300, ge=1)
    long_days: int = Field(default=180, ge=1)
    minimum_events: int = Field(default=60, ge=1)
    minimum_mode_events: int = Field(default=30, ge=1)
    minimum_blocks: int = Field(default=20, ge=2)
    block_days: int = Field(default=1, ge=1)
    window_stride: int = Field(default=60, ge=1)
    maximum_snapshot_age_hours: int = Field(default=24, ge=1, le=24)
    minimum_coverage: float = Field(default=0.95, gt=0, le=1)
    minimum_improvement: float = Field(default=0.05, ge=0, lt=1)
    worst_mode_tolerance: float = Field(default=0.05, ge=0)
    consecutive_windows: int = Field(default=2, ge=1)
    cooldown_days: int = Field(default=14, ge=0)
    cooldown_labels: int = Field(default=60, ge=1)
    bootstrap_replicates: int = Field(default=1000, ge=100, le=10000)
    maximum_p95_ms: float = Field(default=100, gt=0)
    seed: int = 20260911

    @property
    def policy_hash(self):
        return digest(self.model_dump())


def block_interval(differences, decisions, policy):
    days = pd.to_datetime(decisions, utc=True).tz_convert("Asia/Shanghai").normalize().asi8
    groups = days // (86400 * 10**9 * policy.block_days)
    frame = pd.DataFrame({"block": groups, "delta": differences})
    aggregates = frame.groupby("block").delta.agg(["sum", "count"])
    if len(aggregates) < policy.minimum_blocks:
        return {"low": None, "high": None, "blocks": len(aggregates), "status": "INSUFFICIENT_BLOCKS"}
    rng = np.random.default_rng(policy.seed)
    choices = rng.integers(0, len(aggregates), (policy.bootstrap_replicates, len(aggregates)))
    means = aggregates["sum"].to_numpy()[choices].sum(axis=1) / aggregates["count"].to_numpy()[choices].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "low": float(lo),
        "high": float(hi),
        "blocks": len(aggregates),
        "status": "COMPUTED",
        "assumption": "dependence retained inside fixed calendar blocks; between-block independence is not established",
    }


class MatureMetrics:
    def __init__(self, forecasts, labels, attributes, methods, scales, profiles, policy=None):
        self.policy = policy or RoutingPolicy()
        self.methods = tuple(methods)
        if "Persistence" not in self.methods:
            raise WorkbenchError("路由回放必须包含 Persistence", "ROUTING_BASELINE")
        self.scales, self.profiles = scales, profiles
        self.attributes = attributes
        self.events = sorted(attributes, key=lambda e: (source_time(attributes[e]["decision_at"]), e))
        self.position = {e: i for i, e in enumerate(self.events)}
        self.times = [source_time(attributes[e]["decision_at"]) for e in self.events]
        self.chains = defaultdict(list)
        for record in labels.records:
            if record.event_id in attributes:
                self.chains[(record.event_id, record.target)].append(record)
        self.predictions = np.full((len(self.events), len(self.methods), 2), np.nan)
        self.forecast_ids = {}
        expected = {(e, m) for e in self.events for m in self.methods}
        if (
            forecasts.duplicated(["event_id", "method_id"]).any()
            or set(zip(forecasts.event_id, forecasts.method_id, strict=True)) != expected
        ):
            raise WorkbenchError("预测必须保留完整方法×事件账本", "ROUTING_COVERAGE")
        for row in forecasts.itertuples(index=False):
            i, j = self.position[row.event_id], self.methods.index(row.method_id)
            if (
                source_time(row.decision_at) != self.times[i]
                or source_time(row.fit_cutoff_at) > self.times[i]
                or row.status not in ("completed", "failed")
            ):
                raise WorkbenchError("预测的时间截止或状态无效", "ROUTING_PREDICTION_TIME")
        # Array assembly by explicit column names avoids the Python keyword 'as'.
        for method_index, method in enumerate(self.methods):
            rows = forecasts[forecasts.method_id.eq(method)].set_index("event_id").loc[self.events]
            values = rows[["cu", "as"]].to_numpy(float)
            values[~rows.status.eq("completed").to_numpy()] = np.nan
            self.predictions[:, method_index] = values
            self.forecast_ids[method] = [
                digest(
                    {
                        "event_id": e,
                        "method": method,
                        "values": values[i].tolist(),
                        "fit_cutoff_at": rows.loc[e, "fit_cutoff_at"],
                    }
                )
                for i, e in enumerate(self.events)
            ]
        self.valid = np.isfinite(self.predictions).all(axis=2)
        initial = []
        for event in self.events:
            chains = [self.chains[(event, t)] for t in ("cu", "as")]
            if all(chain and chain[0].quality_eligible and chain[0].value is not None for chain in chains):
                initial.append(max(chain[0].available_at.timestamp() for chain in chains))
        self.maturity_times = np.sort(initial)
        self.revision_times = np.sort(
            [r.available_at.timestamp() for chain in self.chains.values() for r in chain if r.revision > 1]
        )

    def visible(self, as_of):
        cutoff = source_time(as_of)
        current = {}
        for key, chain in self.chains.items():
            allowed = [r for r in chain if r.available_at <= cutoff]
            if allowed:
                current[key] = max(allowed, key=lambda r: r.revision)
        return current

    def _eligible(self, visible, cutoff):
        return [
            e
            for e, t in zip(self.events, self.times, strict=True)
            if t < cutoff
            and all(
                (e, target) in visible
                and visible[(e, target)].quality_eligible
                and visible[(e, target)].value is not None
                for target in ("cu", "as")
            )
        ]

    def mature_count(self, as_of):
        return int(np.searchsorted(self.maturity_times, source_time(as_of).timestamp(), side="right"))

    def revision_token(self, as_of):
        return int(np.searchsorted(self.revision_times, source_time(as_of).timestamp(), side="right"))

    def _window(self, eligible, visible, cutoff, count, days):
        p = self.policy
        base = [e for e in eligible if source_time(self.attributes[e]["decision_at"]) > cutoff - timedelta(days=days)][
            -count:
        ]
        positions = [self.position[e] for e in base]
        coverage, excluded, pool = {}, {}, []
        for j, method in enumerate(self.methods):
            n = int(self.valid[positions, j].sum())
            value = n / len(base) if base else 0.0
            coverage[method] = {
                "eligible_events": len(base),
                "valid_predictions": n,
                "missing_predictions": len(base) - n,
                "fraction": value,
            }
            profile = self.profiles.get(method)
            if profile is None or profile.get("p95_ms") is None or profile["p95_ms"] > p.maximum_p95_ms:
                excluded[method] = "INFERENCE_BUDGET"
            elif value < p.minimum_coverage:
                excluded[method] = "PREDICTION_COVERAGE"
            else:
                pool.append(method)
        common = (
            [e for e in base if all(self.valid[self.position[e], self.methods.index(m)] for m in pool)] if pool else []
        )
        positions = [self.position[e] for e in common]
        truth = {t: np.array([visible[(e, t)].value for e in common]) for t in ("cu", "as")}
        metrics = {}
        for method in pool:
            j = self.methods.index(method)
            metrics[method] = {}
            for target_index, target in enumerate(("cu", "as")):
                predicted = self.predictions[positions, j, target_index]
                groups = {}
                for grouping in ("mode", "interval_group", "quality"):
                    grouped = {}
                    for value in sorted({self.attributes[e][grouping] for e in common}):
                        mask = np.array([self.attributes[e][grouping] == value for e in common])
                        grouped[value] = numerical_metrics(truth[target][mask], predicted[mask])
                    groups[grouping] = grouped
                metrics[method][target] = (
                    {**numerical_metrics(truth[target], predicted), "groups": groups} if common else None
                )
        revisions = [visible[(e, t)].record_id for e in common for t in ("cu", "as")]
        return {
            "event_ids": common,
            "common_event_hash": digest(common),
            "label_revision_hash": digest(revisions),
            "label_record_ids": revisions,
            "window_eligible_events": len(base),
            "n": len(common),
            "coverage": coverage,
            "excluded": excluded,
            "eligible_methods": pool,
            "metrics": metrics,
            "start": (cutoff - timedelta(days=days)).isoformat(),
            "forecast_hash": digest({m: [self.forecast_ids[m][self.position[e]] for e in common] for m in pool}),
        }

    def snapshot(self, as_of, mode=None):
        cutoff = source_time(as_of)
        visible = self.visible(cutoff)
        eligible = self._eligible(visible, cutoff)
        p = self.policy
        short = self._window(eligible, visible, cutoff, p.short_events, p.short_days)
        long = self._window(eligible, visible, cutoff, p.long_events, p.long_days)
        blocks = (
            block_interval(np.zeros(long["n"]), [self.attributes[e]["decision_at"] for e in long["event_ids"]], p)
            if long["n"]
            else {"blocks": 0}
        )
        mode_n = sum(self.attributes[e]["mode"] == mode for e in long["event_ids"]) if mode is not None else long["n"]
        enough = (
            short["n"] >= p.minimum_events
            and long["n"] >= p.minimum_events
            and mode_n >= p.minimum_mode_events
            and blocks["blocks"] >= p.minimum_blocks
        )
        snapshot = safe(
            {
                "schema_version": "metric-snapshot.g5a.v1",
                "evaluation_mode": "historical_replay",
                "comparison_unit": "fixed_method_training_policy_across_predeclared_outer_folds",
                "as_of": cutoff.isoformat(),
                "policy_hash": p.policy_hash,
                "window_basis": "origin_decision_at",
                "status": "READY" if enough else "INSUFFICIENT_LABELS",
                "mode": mode,
                "mode_events": mode_n,
                "calendar_blocks": blocks["blocks"],
                "block_days": p.block_days,
                "mature_events": len(eligible),
                "short": short,
                "long": long,
            }
        )
        snapshot["id"] = digest(snapshot)
        return snapshot

    def score(self, snapshot, target, method, mode):
        metric = snapshot["long"]["metrics"][method][target]
        scale = self.scales[target]
        if not math.isfinite(scale) or scale <= 0:
            raise WorkbenchError("训练目标尺度为零，不能计算归一化评分", "ROUTING_SCALE")
        mode_metric = metric["groups"]["mode"].get(mode) if mode else None
        error = (
            mode_metric["mae"] if mode_metric and mode_metric["n"] >= self.policy.minimum_mode_events else metric["mae"]
        )
        supported = [v for v in metric["groups"]["mode"].values() if v["n"] >= self.policy.minimum_mode_events]
        worst = max([v["mae"] for v in supported], default=metric["mae"])
        components = {
            "error": error / scale,
            "worst_mode": worst / scale,
            "absolute_bias": abs(metric["bias"]) / scale,
            "latency": min(1.0, self.profiles[method]["p95_ms"] / self.policy.maximum_p95_ms),
            "api_cost": 0.0,
        }
        weights = {"error": 0.55, "worst_mode": 0.15, "absolute_bias": 0.10, "latency": 0.10, "api_cost": 0.10}
        return {"total": sum(components[k] * weights[k] for k in weights), "components": components, "weights": weights}

    def paired_gate(self, snapshot, target, challenger, champion):
        p = self.policy
        short, long = snapshot["short"], snapshot["long"]
        for name, window in (("short", short), ("long", long)):
            old, new = window["metrics"][champion][target]["mae"], window["metrics"][challenger][target]["mae"]
            if old <= 0 or new > old * (1 - p.minimum_improvement):
                return {
                    "passed": False,
                    "reason": "MINIMUM_IMPROVEMENT",
                    "window": name,
                    "old_mae": old,
                    "new_mae": new,
                }
        old_groups = long["metrics"][champion][target]["groups"]["mode"]
        new_groups = long["metrics"][challenger][target]["groups"]["mode"]
        for mode, old in old_groups.items():
            if old["n"] >= p.minimum_mode_events and new_groups[mode]["mae"] > old["mae"] * (
                1 + p.worst_mode_tolerance
            ):
                return {"passed": False, "reason": "MODE_DEGRADATION", "mode": mode}
        visible = self.visible(snapshot["as_of"])
        positions = [self.position[e] for e in long["event_ids"]]
        actual = np.array([visible[(e, target)].value for e in long["event_ids"]])
        t = ("cu", "as").index(target)
        errors = np.abs(self.predictions[positions, self.methods.index(challenger), t] - actual) - np.abs(
            self.predictions[positions, self.methods.index(champion), t] - actual
        )
        interval = block_interval(errors, [self.attributes[e]["decision_at"] for e in long["event_ids"]], p)
        return {
            "passed": interval["high"] is not None and interval["high"] < 0,
            "reason": "PAIRED_INTERVAL",
            "paired_mae_difference": float(errors.mean()),
            "ci95": interval,
        }
