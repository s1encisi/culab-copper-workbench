from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from copper_mvp.common import WorkbenchError, digest, safe
from copper_mvp.contracts import RunRequest
from copper_mvp.data import DataRepository
from copper_mvp.modeling import ModelManager


def nondominated(F: np.ndarray) -> np.ndarray:
    return NonDominatedSorting().do(np.asarray(F, dtype=float), only_non_dominated_front=True)


def positive_scale(values: np.ndarray) -> np.ndarray:
    scale = np.nanquantile(values, 0.75, axis=0) - np.nanquantile(values, 0.25, axis=0)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


class CandidateProblem(Problem):
    def __init__(self, lower: np.ndarray, upper: np.ndarray, evaluate: Callable, n_constraints: int):
        super().__init__(n_var=len(lower), n_obj=2, n_ieq_constr=n_constraints, xl=lower, xu=upper)
        self.evaluate_values = evaluate
        self.all_x: list[np.ndarray] = []
        self.all_f: list[np.ndarray] = []
        self.all_g: list[np.ndarray] = []

    def _evaluate(self, X, out, *args, **kwargs):
        F, G, _ = self.evaluate_values(np.asarray(X, dtype=float))
        if not np.isfinite(F).all() or not np.isfinite(G).all():
            raise WorkbenchError("候选评价出现非有限数值", "NONFINITE_CANDIDATE")
        self.all_x.append(np.array(X, copy=True)); self.all_f.append(F.copy()); self.all_g.append(G.copy())
        out["F"] = F; out["G"] = G


def solve(data: DataRepository, models: ModelManager, request: RunRequest, output: Path, progress: Callable) -> dict:
    started = perf_counter()
    warnings = []
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
            G = np.column_stack(((y[:, 1] - ref_y[1] - request.epsilon_as) / scale_y[1], -y[:, 0] / scale_y[0], -y[:, 1] / scale_y[1]))
            return np.column_stack((y[:, 0], power)), G, {"currents": currents, "y": y}
        reference_x = base_currents[active_positions][None, :]
        ref_F = np.array([[ref_y[0], ref_power]])
        ref_G = np.array([[-request.epsilon_as / scale_y[1], -ref_y[0] / scale_y[0], -ref_y[1] / scale_y[1]]])
        constraints = 3
    if len(lower):
        problem = CandidateProblem(lower, upper, evaluate, constraints)
        algorithm = NSGA2(pop_size=64, crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=1.0, prob_var=1 / len(lower), eta=20), eliminate_duplicates=True)
        algorithm.setup(problem, termination=("n_eval", request.evaluation_budget), seed=request.seed, verbose=False)
        stop = "evaluation_budget"
        while algorithm.has_next():
            left = request.evaluation_budget - algorithm.evaluator.n_eval
            if left <= 0:
                break
            if algorithm.is_initialized:
                algorithm.n_offsprings = min(64, left)
            algorithm.next()
            progress("候选搜索", {"evaluations": int(algorithm.evaluator.n_eval), "budget": request.evaluation_budget}, 0)
            if perf_counter() - started > 30:
                stop = "time_budget"
                break
        all_x = np.vstack(problem.all_x); all_f = np.vstack(problem.all_f); all_g = np.vstack(problem.all_g)
    else:
        all_x = np.empty((1, 0)); all_f, all_g, _ = evaluate(all_x)
        warnings.append("当前局部范围没有可变化电流，返回参考方案")
        stop = "fixed_reference"
    feasible = np.where(np.all(all_g <= 1e-8, axis=1))[0]
    if len(feasible):
        chosen = feasible[nondominated(all_f[feasible])]
        _, unique = np.unique(np.round(all_f[chosen], 10), axis=0, return_index=True)
        chosen = chosen[np.sort(unique)]
        chosen = chosen[np.argsort(all_f[chosen, 0])]
        final_x = all_x[chosen]
        audited_f, audited_g, details = evaluate(final_x)
        if not np.allclose(audited_f, all_f[chosen], rtol=1e-9, atol=1e-8) or not np.all(audited_g <= 1e-8) or not np.isfinite(audited_f).all():
            raise WorkbenchError("候选复算未通过", "CANDIDATE_AUDIT")
        if len(lower) and (np.any(final_x < lower - 1e-8) or np.any(final_x > upper + 1e-8)):
            raise WorkbenchError("候选越过电流范围", "CANDIDATE_BOUNDS")
    else:
        final_x = np.empty((0, len(lower))); audited_f = np.empty((0, 2)); audited_g = np.empty((0, constraints)); details = {}
    candidates = []
    for i, (point, objectives, constraints_row) in enumerate(zip(final_x, audited_f, audited_g)):
        candidate = {"id": "candidate_" + digest(point.tolist())[:16], "f1": objectives[0], "f2": objectives[1], "constraint_violation": float(max(0, constraints_row.max())), "delta_cu": ref_F[0, 0] - objectives[0], "delta_power": ref_F[0, 1] - objectives[1], "audit_passed": True}
        if request.mode == "benchmark":
            candidate.update({"variables": {"u1": point[0], "u2": point[1]}, "as": None, "as_margin": None, "support_distance": None, "improvement_f1": candidate["delta_cu"], "improvement_f2": candidate["delta_power"], "delta_cu": None, "delta_power": None})
        else:
            currents = details["currents"][i]
            distance = np.min(np.linalg.norm((historical_currents - currents) / current_scale, axis=1)) if len(historical_currents) else None
            candidate.update({"variables": {f"stage{s}_current_a": currents[j] for j, s in enumerate(stages)}, "as": details["y"][i, 1], "as_margin": ref_y[1] + request.epsilon_as - details["y"][i, 1], "support_distance": distance})
        candidates.append(safe(candidate))
    representatives = {}
    if candidates:
        normalized = (audited_f - audited_f.min(axis=0)) / np.where(np.ptp(audited_f, axis=0) > 0, np.ptp(audited_f, axis=0), 1)
        balanced = np.max(0.5 * normalized, axis=1).argmin()
        indices = {"quality": int(audited_f[:, 0].argmin()), "power": int(audited_f[:, 1].argmin()), "balanced": int(balanced)}
        representatives = {k: candidates[i]["id"] for k, i in indices.items()}
        retain = sorted(set(np.linspace(0, len(candidates) - 1, min(128, len(candidates))).astype(int).tolist() + list(indices.values())))
        pd.DataFrame([{"mode": request.mode, **{k: v for k, v in c.items() if k != "variables"}, **c["variables"]} for c in candidates]).to_csv(output / "pareto_candidates.csv", index=False)
        displayed = [candidates[i] for i in retain]
    else:
        displayed = []
    reference = {"f1": ref_F[0, 0], "f2": ref_F[0, 1], "feasible": bool(np.all(ref_G <= 1e-8))}
    if request.mode == "plant":
        reference.update({"as": ref_y[1], "variables": {f"stage{s}_current_a": base_currents[j] for j, s in enumerate(stages)}})
    else:
        reference.update({"as": None, "variables": {"u1": 1.0, "u2": 1.0}})
    return safe({"kind": "optimization", "mode": request.mode, "event_id": request.event_id if context else None, "decision_at": context["decision_at"] if context else None, "model_scope": request.model_scope if context else "mathematical_test", "fold_id": resolved["fold_id"] if resolved else None, "bundle_id": resolved["manifest"]["bundle_id"] if resolved else None, "models": resolved["names"] if resolved else {"function": "constrained_quadratic"}, "objective_labels": labels, "objective_units": units, "ranges": ranges, "supported_stages": stages, "reference": reference, "candidates": displayed, "total_front_points": len(candidates), "representatives": representatives, "search_evaluations": len(all_x), "audit_evaluations": len(final_x), "reference_evaluations": 1, "feasible_evaluations": len(feasible), "feasible_rate": len(feasible) / len(all_x), "elapsed_ms": (perf_counter() - started) * 1000, "stop_reason": stop, "seed": request.seed, "epsilon_as": request.epsilon_as, "warnings": warnings, "solution_status": "feasible_set" if candidates else "no_feasible_candidate_found", "audit": {"passed": True, "checks": ["目标与约束已复算", "变量在本次范围内", "候选为有限数值", "返回非支配候选"]}, "note": "当前电压固定的局部功率代理；候选用于同工况状态探索" if context else "无量纲数学测试；不代表工厂结果"})
