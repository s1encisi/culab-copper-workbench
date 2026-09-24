"""Train-only meta selection and delayed-label adaptive convex weights."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from copper_mvp.common import digest
from copper_mvp.data_contracts import source_time
from copper_mvp.ensemble_models import expand_weights, make_meta, simplex_weights
from copper_mvp.ensemble_time import select_with_tolerance


def meta_features(P, current, subset, target):
    return P[:, list(subset), target] - current[:, [target]]


class AdaptiveTarget:
    def __init__(self, book, target, parameters, initial_weights, scale, history, settings):
        self.book, self.target, self.parameters = book, target, parameters
        self.weights = np.asarray(initial_weights, float).copy()
        self.initial_weights = self.weights.copy()
        self.scale, self.settings = scale, settings
        self.history = list(history)
        self.last_key = None

    def predict(self, event, predictions, *, evidence=True):
        cutoff = self.book.decision(event)
        start = cutoff - timedelta(days=self.settings.dynamic_max_days)
        subset = tuple(self.parameters["subset"])
        self.history = [row for row in self.history if self.book.decision(row[0]) > start]
        eligible = []
        for previous, values in self.history:
            label = self.book.record(previous, ("cu", "as")[self.target], cutoff)
            if label and np.isfinite(values[list(subset)]).all():
                eligible.append((previous, values, label))
        eligible = eligible[-self.parameters["window_events"] :]
        key = digest([row[2].record_id for row in eligible])
        enough = len(eligible) >= self.settings.dynamic_minimum
        if enough and key != self.last_key:
            P = np.array([row[1][list(subset)] for row in eligible])
            y = np.array([row[2].value for row in eligible])
            w = simplex_weights(P, y, self.scale, self.parameters["penalty"], self.weights[list(subset)])
            self.weights = expand_weights(w, subset)
            self.last_key = key
        elif not enough:
            self.weights = self.initial_weights.copy()
            self.last_key = None
        mode = self.book.mode(event)
        local = [row for row in eligible if self.book.mode(row[0]) == mode]
        weights = self.weights.copy()
        strength = 0.0
        if enough and local:
            local_w = simplex_weights(
                np.array([row[1][list(subset)] for row in local]),
                np.array([row[2].value for row in local]),
                self.scale,
                self.parameters["penalty"],
                self.weights[list(subset)],
            )
            strength = len(local) / (len(local) + self.settings.mode_shrinkage)
            weights = (1 - strength) * self.weights + strength * expand_weights(local_w, subset)
        result = float(np.asarray(predictions) @ weights)
        detail = None
        if evidence:
            detail = {
                "schema_version": "adaptive-weights.g5b.v1",
                "evaluation_mode": "historical_replay",
                "event_id": event,
                "target": ("cu", "as")[self.target],
                "as_of": cutoff.isoformat(),
                "parameters": self.parameters,
                "status": "MATURE_WINDOW" if enough else "INITIAL_OOF_WEIGHTS",
                "mature_events": len(eligible),
                "mode_events": len(local),
                "mode": mode,
                "local_strength": strength,
                "event_ids": [row[0] for row in eligible],
                "label_record_ids": [row[2].record_id for row in eligible],
                "cohort_hash": key,
                "global_weights": self.weights.tolist(),
                "weights": weights.tolist(),
                "latest_label_available_at": max((row[2].available_at for row in eligible), default=None),
            }
            if detail["latest_label_available_at"]:
                detail["latest_label_available_at"] = detail["latest_label_available_at"].isoformat()
            detail["id"] = digest(detail)
        # Only forecasts already produced can enter later error windows.
        self.history.append((event, np.asarray(predictions).copy()))
        return result, detail


def choose_meta(book, P, current, events, meta_train, meta_valid, meta_cutoff, outer_cutoff, settings, scale):
    positions = {e: i for i, e in enumerate(events)}
    a = [positions[e] for e in meta_train]
    b = [positions[e] for e in meta_valid]
    y_train = book.y(meta_train, meta_cutoff)
    y_valid = book.y(meta_valid, outer_cutoff)
    chosen = {"stacking": [], "convex": [], "adaptive": []}
    candidates = {"stacking": [], "convex": [], "adaptive": []}
    for target in (0, 1):
        stacks = []
        convex = []
        adaptive = []
        for subset in settings.base_subsets:
            for alpha in settings.ridge_alphas:
                meta = make_meta(alpha)
                meta.fit(meta_features(P[a], current[a], subset, target), y_train[:, target] - current[a, target])
                values = current[b, target] + meta.predict(meta_features(P[b], current[b], subset, target))
                stacks.append(
                    {
                        "target": target,
                        "subset": list(subset),
                        "alpha": alpha,
                        "score": float(np.abs(values - y_valid[:, target]).mean()),
                        "validation_events": len(b),
                    }
                )
            for penalty in settings.convex_penalties:
                w = simplex_weights(P[a][:, list(subset), target], y_train[:, target], scale[target], penalty)
                values = P[b][:, list(subset), target] @ w
                convex.append(
                    {
                        "target": target,
                        "subset": list(subset),
                        "penalty": penalty,
                        "score": float(np.abs(values - y_valid[:, target]).mean()),
                        "validation_events": len(b),
                    }
                )
                for window in settings.dynamic_windows:
                    parameters = {"subset": list(subset), "penalty": penalty, "window_events": window}
                    history = [(e, P[positions[e], :, target]) for e in meta_train]
                    mixer = AdaptiveTarget(
                        book, target, parameters, expand_weights(w, subset), scale[target], history, settings
                    )
                    predicted = [mixer.predict(e, P[positions[e], :, target], evidence=False)[0] for e in meta_valid]
                    adaptive.append(
                        {
                            "target": target,
                            **parameters,
                            "score": float(np.abs(np.asarray(predicted) - y_valid[:, target]).mean()),
                            "validation_events": len(b),
                        }
                    )
        chosen["stacking"].append(
            select_with_tolerance(
                stacks, settings.selection_tolerance, lambda r: (len(r["subset"]), -r["alpha"], r["score"])
            )
        )
        chosen["convex"].append(
            select_with_tolerance(
                convex, settings.selection_tolerance, lambda r: (len(r["subset"]), -r["penalty"], r["score"])
            )
        )
        chosen["adaptive"].append(
            select_with_tolerance(
                adaptive,
                settings.selection_tolerance,
                lambda r: (len(r["subset"]), -r["penalty"], -r["window_events"], r["score"]),
            )
        )
        candidates["stacking"].extend(stacks)
        candidates["convex"].extend(convex)
        candidates["adaptive"].extend(adaptive)
    return {
        "selected": chosen,
        "candidates": candidates,
        "meta_selection_train": book.evidence(meta_train, meta_cutoff),
        "meta_selection_validation_ids": meta_valid,
        "hyperparameters_selected_at": source_time(outer_cutoff).isoformat(),
    }
