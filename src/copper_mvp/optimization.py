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
from copper_mvp.optimization_problem import build_problem, positive_scale


def nondominated(F: np.ndarray) -> np.ndarray:
    return NonDominatedSorting().do(np.asarray(F, dtype=float), only_non_dominated_front=True)


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
    prepared = build_problem(data, models, request)
    lower, upper, reference_x = prepared.lower, prepared.upper, prepared.reference_x
    ref_F, ref_G, evaluate = prepared.ref_F, prepared.ref_G, prepared.evaluate
    labels, units, constraints = prepared.labels, prepared.units, prepared.constraints
    warnings, context, resolved = prepared.warnings, prepared.context, prepared.resolved
    stages, variable_stages, ranges = prepared.stages, prepared.variable_stages, prepared.ranges
    ref_y, base_currents = prepared.ref_y, prepared.base_currents
    historical_currents, current_scale = prepared.historical_currents, prepared.current_scale
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
