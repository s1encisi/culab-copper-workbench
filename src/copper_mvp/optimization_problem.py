"""Shared numerical problem construction for legacy and registered optimizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from copper_mvp.common import WorkbenchError, safe
from copper_mvp.contracts import RunRequest
from copper_mvp.data_contracts import source_time


def positive_scale(values):
    scale = np.nanquantile(values, 0.75, axis=0) - np.nanquantile(values, 0.25, axis=0)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


@dataclass
class PreparedProblem:
    request: RunRequest
    lower: np.ndarray
    upper: np.ndarray
    reference_x: np.ndarray
    ref_F: np.ndarray
    ref_G: np.ndarray
    evaluate: Callable
    constraints: int
    labels: list
    units: list
    warnings: list
    context: dict | None
    resolved: dict | None
    stages: list
    variable_stages: list
    ranges: list
    ref_y: np.ndarray | None
    base_currents: np.ndarray | None
    historical_currents: np.ndarray | None
    current_scale: np.ndarray | None
    voltages: np.ndarray | None
    scale_y: np.ndarray | None
    reference_front: np.ndarray | None = None

    def specification(self):
        artifacts = {}
        if self.resolved:
            for target, name in self.resolved["names"].items():
                key = f"{self.resolved['fold_id']}:{name}:{target}"
                artifacts[target] = self.resolved["manifest"]["artifacts"][key]["sha256"]
        return safe({"schema_version": "problem-spec.g2b.v1", "mode": self.request.mode,
            "event_id": self.context["event_id"] if self.context else None,
            "decision_at": source_time(self.context["decision_at"]).isoformat() if self.context else None,
            "dataset_version": self.context["dataset_version"] if self.context else None,
            "variables": [{"name": f"stage{s}_current_a", "unit": "A", "lower": float(lo), "upper": float(hi)}
                          for s, lo, hi in zip(self.variable_stages, self.lower, self.upper)] if self.context else
                         [{"name": f"u{i+1}", "unit": "dimensionless", "lower": float(lo), "upper": float(hi)}
                          for i, (lo, hi) in enumerate(zip(self.lower, self.upper))],
            "objectives": [{"name": name, "unit": unit, "direction": "minimize"} for name, unit in zip(self.labels, self.units)],
            "constraints": {"kind": "as_allowance_and_nonnegative_predictions" if self.context else "u1+u2<=3",
                            "epsilon_as_mg_l": self.request.epsilon_as if self.context else None,
                            "scales": self.scale_y.tolist() if self.context else None, "normalized_tolerance": 1e-8},
            "fixed_context": {"stages": self.stages, "reference_currents": self.base_currents.tolist(),
                              "fixed_voltages": self.voltages.tolist()} if self.context else {},
            "reference": {"x": self.reference_x[0].tolist(), "F": self.ref_F[0].tolist(), "G": self.ref_G[0].tolist(),
                          "as": float(self.ref_y[1]) if self.context else None},
            "model_scope": self.request.model_scope if self.context else "mathematical_test",
            "fold_id": self.resolved["fold_id"] if self.resolved else None,
            "bundle_id": self.resolved["manifest"]["bundle_id"] if self.resolved else None,
            "model_artifact_hashes": artifacts, "ranges": self.ranges,
            "result_kind": "conditional_model_scenario" if self.context else "mathematical_benchmark",
            "execution_authorized": False})


def build_problem(data, models, request: RunRequest):
    warnings = []
    ref_y = base_currents = historical_currents = current_scale = voltages = scale_y = None
    epsilon_as = request.epsilon_as
    reference_front = None
    context = None; resolved = None; stages = []; variable_stages = []; ranges = []
    if request.mode == "benchmark":
        lower = np.zeros(2); upper = np.full(2, 3.0)
        reference_x = np.ones((1, 2))
        labels = ["目标 f₁", "目标 f₂"]
        units = ["无量纲", "无量纲"]
        def evaluate(X):
            return np.column_stack(((X * X).sum(axis=1), ((X - 2) ** 2).sum(axis=1))), (X.sum(axis=1) - 3)[:, None], {"variables": X}
        ref_F, ref_G, _ = evaluate(reference_x)
        constraints = 1
        t = np.linspace(0, 1.5, 1001)
        reference_front = np.column_stack((2 * t**2, 2 * (t - 2)**2))
    else:
        if request.model_profile == "Persistence":
            raise WorkbenchError("优化需要响应模型，请选择自动选择或已训练的变化量模型", "RESPONSE_MODEL_REQUIRED")
        context = data.context(request.event_id)
        stages = context["supported_stages"]
        if not stages:
            raise WorkbenchError("该事件缺少配对的正电流与电压，请选择另一个事件", "ELECTRICAL_DATA_MISSING")
        resolved = models.resolve(request.event_id, request.model_profile, request.model_scope, request.bundle_id)
        fold = None if resolved["fold_id"] == "DEVELOPMENT" else resolved["fold_id"]
        train_ids = data.training_ids(fold)
        same = [e for e in train_ids if data.mode_cards[e]["mode_code"] == context["mode_code"]]
        history_ids = same if len(same) >= 30 else train_ids
        if len(same) < 30:
            warnings.append("同工况训练样本少于30条，电流范围使用对应训练期的正电流统计")
        if len(stages) == 1:
            warnings.append(f"电功率代理覆盖三四段中的第{stages[0]}段")
        row = data.row(request.event_id)
        base_currents = np.array([row[f"stage{s}_current_a__t_minus_0h"] for s in stages], dtype=float)
        voltages = np.array([row[f"stage{s}_voltage_v__t_minus_0h"] for s in stages], dtype=float)
        lower_list = []; upper_list = []; active_positions = []
        for j, stage in enumerate(stages):
            series = data.frame.loc[history_ids, f"stage{stage}_current_a__t_minus_0h"].to_numpy(float)
            values = series[np.isfinite(series) & (series > 0)]
            if len(values) < 2:
                warnings.append(f"第{stage}段历史电流不足，本次固定该段")
                continue
            p5, p95 = np.quantile(values, [0.05, 0.95])
            lo = max(p5, (1 - request.radius) * base_currents[j])
            hi = min(p95, (1 + request.radius) * base_currents[j])
            varies = hi - lo > 1e-6
            ranges.append({"stage": stage, "current": base_currents[j], "lower": lo if varies else base_currents[j], "upper": hi if varies else base_currents[j], "historical_p5": p5, "historical_p95": p95, "historical_n": len(values), "varies": varies, "source": "same_mode" if len(same) >= 30 else "training_period"})
            if varies:
                variable_stages.append(stage); active_positions.append(j); lower_list.append(lo); upper_list.append(hi)
        lower = np.asarray(lower_list); upper = np.asarray(upper_list)
        base_matrix = data.candidate_matrix(request.event_id, stages, base_currents[None, :])
        ref_y = models.predict_matrix(resolved, base_matrix)[0]
        ref_power = float(base_currents @ voltages / 1000)
        _, train_y = data.training_data()
        scale_y = positive_scale(train_y.loc[train_ids].to_numpy(float))
        historical_currents = data.frame.loc[history_ids, [f"stage{s}_current_a__t_minus_0h" for s in stages]].to_numpy(float)
        historical_currents = historical_currents[np.isfinite(historical_currents).all(axis=1) & (historical_currents > 0).all(axis=1)]
        current_scale = positive_scale(historical_currents) if len(historical_currents) else np.ones(len(stages))
        labels = ["预测 Cu", "电功率代理"]
        units = ["g/L", "kW"]
        def evaluate(X):
            currents = np.repeat(base_currents[None, :], len(X), axis=0)
            for j, position in enumerate(active_positions):
                currents[:, position] = X[:, j]
            matrix = data.candidate_matrix(request.event_id, stages, currents)
            y = models.predict_matrix(resolved, matrix)
            power = currents @ voltages / 1000
            G = np.column_stack(((y[:, 1] - ref_y[1] - epsilon_as) / scale_y[1], -y[:, 0] / scale_y[0], -y[:, 1] / scale_y[1]))
            return np.column_stack((y[:, 0], power)), G, {"currents": currents, "y": y}
        reference_x = base_currents[active_positions][None, :]
        ref_F = np.array([[ref_y[0], ref_power]])
        ref_G = np.array([[-epsilon_as / scale_y[1], -ref_y[0] / scale_y[0], -ref_y[1] / scale_y[1]]])
        constraints = 3
    return PreparedProblem(request=request, lower=lower, upper=upper, reference_x=reference_x,
        ref_F=ref_F, ref_G=ref_G, evaluate=evaluate, constraints=constraints, labels=labels, units=units,
        warnings=warnings, context=context, resolved=resolved, stages=stages, variable_stages=variable_stages,
        ranges=ranges, ref_y=ref_y, base_currents=base_currents, historical_currents=historical_currents,
        current_scale=current_scale, voltages=voltages, scale_y=scale_y, reference_front=reference_front)
