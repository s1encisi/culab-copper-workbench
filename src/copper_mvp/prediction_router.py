"""Deterministic shadow routing; no live champion or model artifact is changed."""

from __future__ import annotations

from datetime import timedelta

from copper_mvp.common import digest
from copper_mvp.data_contracts import source_time


class PolicyRouter:
    def __init__(self, metrics, *, by_mode, on_snapshot=lambda snapshot: None):
        self.metrics, self.by_mode, self.on_snapshot = metrics, by_mode, on_snapshot
        self.states = {}
        self.evidence = {}

    def decide(self, event_id):
        engine, p = self.metrics, self.metrics.policy
        attribute = engine.attributes[event_id]
        at = source_time(attribute["decision_at"])
        mode = attribute["mode"] if self.by_mode else None
        key = mode or "*"
        maturity, revisions = engine.mature_count(at), engine.revision_token(at)
        previous = self.evidence.get(key)
        distinct = previous is None or maturity - previous["confirmation_count"] >= p.window_stride
        refresh = (
            distinct
            or previous["revisions"] != revisions
            or at - source_time(previous["snapshot"]["as_of"]) >= timedelta(hours=p.maximum_snapshot_age_hours)
        )
        if previous and previous["revisions"] != revisions:
            for (context, target), state in self.states.items():
                if context == key:
                    state.update(streak=0, challenger=None, confirmation_snapshots=[])
        if refresh:
            snapshot = engine.snapshot(at, mode)
            self.on_snapshot(snapshot)
            self.evidence[key] = {
                "snapshot": snapshot,
                "revisions": revisions,
                "confirmation_count": maturity if distinct else previous["confirmation_count"],
            }
        else:
            snapshot = previous["snapshot"]
        decisions = {}
        for target_index, target in enumerate(("cu", "as")):
            state = self.states.setdefault(
                (key, target),
                {
                    "champion": "Persistence",
                    "streak": 0,
                    "challenger": None,
                    "confirmation_snapshots": [],
                    "last_switch_time": None,
                    "last_switch_count": 0,
                },
            )
            pool = sorted(set(snapshot["short"]["eligible_methods"]) & set(snapshot["long"]["eligible_methods"]))
            result = {
                "event_id": event_id,
                "target": target,
                "decision_at": at.isoformat(),
                "metric_snapshot_id": snapshot["id"],
                "metric_as_of": snapshot["as_of"],
                "policy_hash": p.policy_hash,
                "mode": mode,
                "candidate_methods": pool,
                "scores": {},
                "new_confirmation_window": distinct,
                "evaluation_mode": "historical_replay",
                "deployment": "shadow_replay",
                "live_model_change": False,
            }
            choice, reason = state["champion"], "KEEP_CHAMPION"
            if snapshot["status"] != "READY" or "Persistence" not in pool:
                choice, reason = "Persistence", "INSUFFICIENT_LABELS"
                if distinct or refresh:
                    state.update(streak=0, challenger=None, confirmation_snapshots=[])
            elif choice not in pool:
                state.update(champion="Persistence", streak=0, challenger=None, confirmation_snapshots=[])
                choice, reason = "Persistence", "CHAMPION_INELIGIBLE"
            if snapshot["status"] == "READY" and "Persistence" in pool:
                scores = {m: engine.score(snapshot, target, m, mode) for m in pool}
                challenger = min(pool, key=lambda m: (scores[m]["total"], m))
                result["scores"] = scores
                if challenger != choice:
                    gate = engine.paired_gate(snapshot, target, challenger, choice)
                    result["switch_gate"] = gate
                    if distinct:
                        history = state["confirmation_snapshots"] if state["challenger"] == challenger else []
                        state["confirmation_snapshots"] = (
                            (history + [snapshot["id"]])[-p.consecutive_windows :] if gate["passed"] else []
                        )
                        state["streak"] = len(state["confirmation_snapshots"])
                        state["challenger"] = challenger if gate["passed"] else None
                    result["confirmation_snapshots"] = list(state["confirmation_snapshots"])
                    result["confirmation_windows"] = state["streak"]
                    elapsed = (
                        (at - source_time(state["last_switch_time"])).total_seconds() / 86400
                        if state["last_switch_time"]
                        else float("inf")
                    )
                    cooldown_ok = (
                        elapsed >= p.cooldown_days or maturity - state["last_switch_count"] >= p.cooldown_labels
                    )
                    if (
                        gate["passed"]
                        and state["challenger"] == challenger
                        and state["streak"] >= p.consecutive_windows
                        and cooldown_ok
                    ):
                        choice, reason = challenger, "SHADOW_SWITCH"
                        state.update(
                            champion=choice,
                            streak=0,
                            challenger=None,
                            confirmation_snapshots=[],
                            last_switch_time=at.isoformat(),
                            last_switch_count=maturity,
                        )
                    else:
                        reason = (
                            "WAIT_CONFIRMATION"
                            if gate["passed"] and cooldown_ok
                            else "COOLDOWN"
                            if gate["passed"]
                            else "KEEP_CHAMPION"
                        )
                elif distinct:
                    state.update(streak=0, challenger=None, confirmation_snapshots=[])
            j = engine.methods.index(choice)
            predicted = engine.predictions[engine.position[event_id], j, target_index]
            if not engine.valid[engine.position[event_id], j] or predicted < 0:
                choice, reason = "Persistence", "CURRENT_PREDICTION_INVALID"
                state.update(champion=choice, streak=0, challenger=None, confirmation_snapshots=[])
                predicted = engine.predictions[engine.position[event_id], engine.methods.index(choice), target_index]
            result.setdefault("confirmation_snapshots", list(state["confirmation_snapshots"]))
            result.setdefault("confirmation_windows", state["streak"])
            result.update(
                selected_method=choice,
                value=float(predicted),
                reason=reason,
                last_switch_time=state["last_switch_time"],
            )
            result["id"] = digest(result)
            decisions[target] = result
        return decisions
