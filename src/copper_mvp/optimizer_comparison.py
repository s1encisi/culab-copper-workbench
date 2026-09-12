"""One F/G evaluator, equal total budgets and independently verified fronts."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.algorithms.moo.spea2 import SPEA2, SPEA2Survival
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
from pymoo.algorithms.moo.sms import SMSEMOA, LeastHypervolumeContributionSurvival
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from threadpoolctl import threadpool_limits

from copper_mvp.common import WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.optimization import nondominated
from copper_mvp.optimization_problem import build_problem
from copper_mvp.optimizer_methods import EXTENDED_OPTIMIZERS, make_extended_optimizer, optimizer_method_spec
from copper_mvp.platypus_methods import PLATYPUS_OPTIMIZERS, PlatypusAdapter, platypus_method_spec
from copper_mvp.scalarization import SCALAR_OPTIMIZERS, ScalarizationAdapter, scalarization_spec

TOLERANCE = 1e-8


class EvaluationService:
    """Count every actual vector evaluation, including reference, pilot and review."""
    def __init__(self, prepared, budget):
        self.prepared = prepared
        self.budget = budget
        self.groups = [{"phase": "reference", "X": prepared.reference_x.copy(),
                        "F": prepared.ref_F.copy(), "G": prepared.ref_G.copy()}]
        self.counts = {"reference": 1, "pilot": 0, "initial": 0, "search": 0, "verification": 0}
        self.started = perf_counter()
        self.first_feasible_ms = 0.0 if (np.all(prepared.ref_G <= TOLERANCE) and np.all(prepared.reference_x >= prepared.lower) and np.all(prepared.reference_x <= prepared.upper)) else None

    @property
    def used(self):
        return sum(self.counts.values())

    def evaluate(self, X, phase):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.prepared.lower):
            raise WorkbenchError("候选维数不匹配", "OPTIMIZER_SHAPE")
        if self.used + len(X) > self.budget:
            raise WorkbenchError("优化总求值预算不足", "OPTIMIZER_BUDGET")
        self.counts[phase] += len(X)
        F, G, details = self.prepared.evaluate(X)
        if not np.isfinite(F).all() or not np.isfinite(G).all():
            raise WorkbenchError("目标或约束出现非有限值", "NONFINITE_CANDIDATE")
        self.groups.append({"phase": phase, "X": X.copy(), "F": F.copy(), "G": G.copy()})
        if self.first_feasible_ms is None and np.any(np.all(G <= TOLERANCE, axis=1)):
            self.first_feasible_ms = (perf_counter() - self.started) * 1000
        return F, G, details

    def search_points(self):
        groups = [g for g in self.groups if g["phase"] in ("pilot", "initial", "search")]
        return tuple(np.vstack([g[key] for g in groups]) for key in ("X", "F", "G"))

    def save(self, output):
        rows = []
        for group in self.groups:
            for X, F, G in zip(group["X"], group["F"], group["G"]):
                rows.append({"phase": group["phase"], **{f"x{i}": x for i, x in enumerate(X)},
                             **{f"f{i}": f for i, f in enumerate(F)}, **{f"g{i}": g for i, g in enumerate(G)}})
        pd.DataFrame(rows).to_csv(output / "evaluations.csv", index=False)


class RegisteredProblem(Problem):
    def __init__(self, evaluator, objective_scale=None):
        self.objective_scale = objective_scale
        prepared = evaluator.prepared
        super().__init__(n_var=len(prepared.lower), n_obj=2, n_ieq_constr=prepared.constraints,
                         xl=prepared.lower, xu=prepared.upper)
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        F, G, _ = self.evaluator.evaluate(X, "search")
        out["F"] = F if self.objective_scale is None else F / self.objective_scale
        out["G"] = G - TOLERANCE


class ConstantSafeNormalization(HyperplaneNormalization):
    def update(self, F, nds=None):
        super().update(F, nds)
        span = self.nadir_point - self.ideal_point
        self.nadir_point = np.where(span > 0, self.nadir_point, self.ideal_point + 1.0)


def fresh_spea_survival():
    survival = SPEA2Survival(normalize=True)
    survival.norm = ConstantSafeNormalization(2)
    return survival


class FixedReferenceSurvival(LeastHypervolumeContributionSurvival):
    def __init__(self, scale):
        super().__init__(eps=0.1)
        self.scale = np.asarray(scale)

    def _do(self, problem, pop, *args, ideal=None, nadir=None, **kwargs):
        return super()._do(problem, pop, *args, ideal=np.zeros(2), nadir=self.scale, **kwargs)


def make_optimizer(name, population, scale, n_var):
    kwargs = {"pop_size": 64, "sampling": population, "crossover": SBX(prob=0.9, eta=15),
              "mutation": PM(prob=1.0, prob_var=1 / n_var, eta=20), "eliminate_duplicates": True}
    if name == "NSGA-II":
        return NSGA2(**kwargs, survival=RankAndCrowding())
    if name == "SPEA2":
        return SPEA2(**kwargs, survival=fresh_spea_survival())
    if name == "SMS-EMOA":
        return SMSEMOA(**kwargs, n_offsprings=1, normalize=False, survival=FixedReferenceSurvival(scale))
    if name in EXTENDED_OPTIMIZERS:
        return make_extended_optimizer(name, population, n_var)
    if name in PLATYPUS_OPTIMIZERS:
        return PlatypusAdapter(name, population, scale)
    if name in SCALAR_OPTIMIZERS:
        return ScalarizationAdapter(name, population, scale)
    raise WorkbenchError("没有该优化器", "OPTIMIZER_NOT_FOUND")


def metric_configuration(prepared, pilot_f):
    if prepared.context is None:
        scale = np.array([18., 8.])  # Fixed bounds of the declared quadratic problem.
        reference_kind = "analytical_objective_box"
    else:
        scale = np.maximum(np.vstack((pilot_f, prepared.ref_F)).max(axis=0), 1.0)
        reference_kind = "fixed_3_point_per_axis_pilot_plus_margin"
    return {"origin": [0., 0.], "scale": scale.tolist(), "reference_point": (scale * 1.1).tolist(),
            "reference_kind": reference_kind, "normalized_reference": [1.1, 1.1]}


def run_optimizer(prepared, optimizer_id, seed, budget, seconds, output, progress=lambda detail: None):
    started = perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "failure.json").exists():
        raise WorkbenchError("该优化运行已有结果或失败记录", "OPTIMIZER_RUN_EXISTS")
    evaluator = EvaluationService(prepared, budget)
    try:
        return _run_optimizer(prepared, optimizer_id, seed, budget, seconds, output, progress, evaluator, started)
    except Exception as exc:
        evaluator.save(output)
        completed_rows = sum(len(group["X"]) for group in evaluator.groups)
        write_json(output / "failure.json", {"status": "failed", "optimizer_id": optimizer_id, "seed": seed,
            "error_code": getattr(exc, "code", type(exc).__name__), "charged_evaluations": evaluator.counts,
            "total_evaluations": evaluator.used, "completed_evaluation_rows": completed_rows,
            "charged_without_saved_values": evaluator.used-completed_rows, "elapsed_ms": (perf_counter()-started)*1000})
        raise


def _run_optimizer(prepared, optimizer_id, seed, budget, seconds, output, progress, evaluator, started):
    dimensions = len(prepared.lower)
    pilot_unit = np.array(list(product((0., 0.5, 1.), repeat=dimensions))).reshape(-1, dimensions) if dimensions else np.empty((1, 0))
    pilot_x = prepared.lower + pilot_unit * (prepared.upper - prepared.lower)
    pilot_f, pilot_g, _ = evaluator.evaluate(pilot_x, "pilot")
    metric = metric_configuration(prepared, pilot_f)
    spec = {**prepared.specification(), "metric": metric, "evaluator_version": "vector-evaluator.g2b.v1",
            "evaluator_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in ("optimization_problem.py", "optimizer_comparison.py", "optimizer_methods.py", "platypus_methods.py", "hype.py", "scalarization.py")},
            "reference_front_hash": digest(prepared.reference_front.tolist()) if prepared.reference_front is not None else None,
            "budget_mode": "total_equivalent_v2", "total_budget": budget, "cache_policy": "disabled_for_comparison"}
    signature = digest(spec)
    reserve = min(128, max(16, budget // 8))
    write_json(output / "problem.json", {"signature": signature, **spec})
    algorithm_executed = bool(dimensions)
    stop = "fixed_reference"
    if dimensions:
        rng = np.random.default_rng(seed)
        initial_x = rng.uniform(prepared.lower, prepared.upper, size=(64 - len(pilot_x), dimensions))
        initial_f, initial_g, _ = evaluator.evaluate(initial_x, "initial")
        X = np.vstack((pilot_x, initial_x)); F = np.vstack((pilot_f, initial_f)); G = np.vstack((pilot_g, initial_g))
        objective_scale = np.asarray(metric["scale"]) if optimizer_id in EXTENDED_OPTIMIZERS + PLATYPUS_OPTIMIZERS + SCALAR_OPTIMIZERS else None
        search_f = F if objective_scale is None else F / objective_scale
        population = Population.new(X=X, F=search_f, G=G - TOLERANCE, H=np.empty((len(X), 0)))
        population.apply(lambda individual: individual.evaluated.update(("F", "G", "H")))
        algorithm = make_optimizer(optimizer_id, population, metric["scale"], dimensions)
        termination = ("n_gen", 1 + int(np.ceil((budget - reserve - evaluator.used) / 64))) if optimizer_id == "RVEA" else ("n_eval", budget)
        algorithm.setup(RegisteredProblem(evaluator, objective_scale), termination=termination, seed=seed, verbose=False)
        if optimizer_id in PLATYPUS_OPTIMIZERS + SCALAR_OPTIMIZERS:
            algorithm.deadline = started + seconds
        algorithm.next()  # Supplied initial values are already evaluated and charged.
        stop = "total_budget"
        last_report = 0
        while algorithm.has_next():
            remaining = budget - reserve - evaluator.used
            if remaining <= 0:
                break
            if perf_counter() - started >= seconds:
                stop = "time_budget"
                break
            algorithm.n_offsprings = remaining if optimizer_id in SCALAR_OPTIMIZERS else min(1 if optimizer_id in ("SMS-EMOA", "MOEA-D") else 64, remaining)
            previous = evaluator.used
            algorithm.next()
            if evaluator.used == previous:
                stop = "no_new_candidates"
                break
            if evaluator.used - last_report >= 64:
                progress({"optimizer": optimizer_id, "seed": seed, "evaluations": evaluator.used, "budget": budget})
                last_report = evaluator.used
        if optimizer_id in PLATYPUS_OPTIMIZERS + SCALAR_OPTIMIZERS and algorithm.stop_reason is not None:
            stop = algorithm.stop_reason
    X, F, G = evaluator.search_points()
    feasible = np.flatnonzero(np.all(G <= TOLERANCE, axis=1))
    front_indices = feasible[nondominated(F[feasible])] if len(feasible) else np.array([], dtype=int)
    decision_front_count = len(np.unique(X[front_indices], axis=0)) if len(front_indices) else 0
    if len(front_indices):
        _, unique = np.unique(np.round(F[front_indices], 10), axis=0, return_index=True)
        front_indices = front_indices[np.sort(unique)]
        front_indices = front_indices[np.argsort(F[front_indices, 0])]
    objective_front_count = len(front_indices)
    if len(front_indices) > reserve:
        front_indices = front_indices[np.unique(np.linspace(0, len(front_indices) - 1, reserve).astype(int))]
    verified_x = X[front_indices]
    if len(front_indices):
        verified_f, verified_g, details = evaluator.evaluate(verified_x, "verification")
        if (not np.allclose(verified_f, F[front_indices], rtol=1e-9, atol=1e-8)
            or not np.allclose(verified_g, G[front_indices], rtol=1e-9, atol=1e-8)
            or not np.all(verified_g <= TOLERANCE)
            or np.any(verified_x < prepared.lower - TOLERANCE) or np.any(verified_x > prepared.upper + TOLERANCE)):
            raise WorkbenchError("独立候选复算不一致", "CANDIDATE_AUDIT")
    else:
        verified_f = np.empty((0, 2)); verified_g = np.empty((0, prepared.constraints)); details = {}
    scale = np.array(metric["scale"]); normalized = verified_f / scale
    inside = np.all(normalized <= 1.1, axis=1)
    hv = float(HV(ref_point=np.array([1.1, 1.1]))(normalized[inside])) if inside.any() else 0.0
    igd = None
    if prepared.reference_front is not None and len(normalized):
        igd = float(IGDPlus(prepared.reference_front / scale)(normalized))
    candidates = []
    for i, (x, f, g) in enumerate(zip(verified_x, verified_f, verified_g)):
        variables = dict(zip([v["name"] for v in spec["variables"]], x.tolist()))
        row = {"id": "candidate_" + digest(x.tolist())[:16], "variables": variables, "f1": f[0], "f2": f[1],
               "constraints": g.tolist(), "audit_passed": True}
        if prepared.context:
            row["as"] = details["y"][i, 1]
            currents = details["currents"][i]
            row["all_currents"] = {f"stage{s}_current_a": currents[j] for j, s in enumerate(prepared.stages)}
            row["support_distance"] = float(np.min(np.linalg.norm((prepared.historical_currents - currents) / prepared.current_scale, axis=1))) if len(prepared.historical_currents) else None
        candidates.append(safe(row))
    result = safe({"schema_version": "optimizer-result.g2b.v1", "problem_signature": signature,
        "optimizer_id": optimizer_id, "seed": seed, "status": "feasible" if candidates else "infeasible",
        "status_reason": "verified_candidates" if candidates else "none_found_within_budget",
        "algorithm_executed": algorithm_executed, "stop_reason": stop, "created_at": utc_now(),
        "budget_mode": spec["budget_mode"], "budget_limit": budget, "evaluations": evaluator.counts,
        "total_evaluations": evaluator.used, "cache_hits": 0, "retries": 0,
        "initial_population_hash": digest(X[:64].tolist()) if dimensions else digest(X.tolist()),
        "metric": metric, "hv": hv, "igd_plus": igd,
        "igd_reference": "declared_analytic_front" if prepared.reference_front is not None else "no_independent_reference_front",
        "outside_reference_box": int((~inside).sum()), "feasible_search_points": len(feasible),
        "feasible_rate": len(feasible) / len(X), "decision_front_points": decision_front_count,
        "objective_front_points": objective_front_count, "verified_front_points": len(candidates),
        "first_feasible_ms": evaluator.first_feasible_ms, "timing_scope": "pilot_initialization_search_and_verification_excludes_problem_assembly", "elapsed_ms": (perf_counter() - started) * 1000,
        "candidates": candidates, "warnings": prepared.warnings, "execution_authorized": False})
    if optimizer_id in EXTENDED_OPTIMIZERS:
        result["optimizer_spec"] = optimizer_method_spec(optimizer_id)
    elif optimizer_id in PLATYPUS_OPTIMIZERS:
        result["optimizer_spec"] = platypus_method_spec(optimizer_id)
    elif optimizer_id in SCALAR_OPTIMIZERS:
        result["optimizer_spec"] = scalarization_spec(optimizer_id)
        result["scalarization"] = algorithm.details() if dimensions else {"status": "fixed_reference", "jobs": []}
    evaluator.save(output)
    write_json(output / "result.json", result)
    return result


def compare_optimizers(data, models, request, output, progress=lambda detail: None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise WorkbenchError("比较目录已存在", "OPTIMIZER_RUN_EXISTS")
    write_json(output / "protocol.json", {"request": request.model_dump(mode="json"), "created_at": utc_now(),
               "selection_policy": "same_context_initial_population_total_budget_and_metric_reference",
               "software": optimizer_software(), "cache_policy": "disabled_for_comparison", "automatic_promotion": False})
    results = []
    with threadpool_limits(limits=1):
        for case, event in enumerate(request.event_ids if request.mode == "plant" else (None,)):
            signatures = set(); initial_hashes = {}
            for seed in request.seeds:
                for optimizer in request.optimizers:
                    progress({"case": case, "optimizer": optimizer, "seed": seed, "phase": "running"})
                    folder = output / f"case_{case}" / str(seed) / optimizer
                    prepared = build_problem(data, models, request.problem_request(event, seed))
                    result = run_optimizer(prepared, optimizer, seed, request.total_budget, request.seconds_per_run, folder, progress)
                    signatures.add(result["problem_signature"])
                    if seed in initial_hashes and initial_hashes[seed] != result["initial_population_hash"]:
                        raise WorkbenchError("比较初始种群不一致", "OPTIMIZER_FAIRNESS")
                    initial_hashes[seed] = result["initial_population_hash"]
                    results.append({"case": case, "event_id": event, **result})
            if len(signatures) != 1:
                raise WorkbenchError("问题、预算或参考点签名不一致", "OPTIMIZER_FAIRNESS")
    summary = {"schema_version": "optimizer-comparison.g2b.v1", "status": "completed",
               "cases": len(request.event_ids) if request.mode == "plant" else 1,
               "optimizers": list(request.optimizers), "seeds": list(request.seeds),
               "runs": len(results), "results": results, "automatic_promotion": False}
    write_json(output / "comparison.json", summary)
    rows = [{k: r[k] for k in ("case", "event_id", "optimizer_id", "seed", "status", "hv", "igd_plus", "total_evaluations",
                             "feasible_rate", "decision_front_points", "objective_front_points", "verified_front_points", "elapsed_ms", "stop_reason", "algorithm_executed")} for r in results]
    for row, result in zip(rows, results):
        detail = result.get("scalarization")
        row["scalarization_converged"] = detail["converged_subproblems"] if detail and "converged_subproblems" in detail else None
        row["scalarization_subproblems"] = len(detail["jobs"]) if detail else None
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    report = "# 注册优化方法比较\n\n"
    report += f"工况 {summary['cases']} 个，优化器 {len(request.optimizers)} 个，种子 {len(request.seeds)} 个。每次总求值上限 {request.total_budget}，包含参考、探测、初始种群、搜索和复算。\n\n"
    report += "| 工况 | 优化器 | 种子 | 候选状态 | 停止原因 | 子问题收敛 | HV | IGD+ | 求值 | 复核前沿点 | 耗时(ms) |\n|---|---|---:|---|---|---|---:|---:|---:|---:|---:|\n"
    for r in results:
        igd_text = f"{r['igd_plus']:.6g}" if r["igd_plus"] is not None else "无独立参考集"
        detail = r.get("scalarization")
        convergence = f"{detail['converged_subproblems']}/{len(detail['jobs'])}" if detail and "converged_subproblems" in detail else "—"
        report += f"| {r['case']} | {r['optimizer_id']} | {r['seed']} | {r['status']} | {r['stop_reason']} | {convergence} | {r['hv']:.6g} | {igd_text} | {r['total_evaluations']} | {r['verified_front_points']} | {r['elapsed_ms']:.1f} |\n"
    report += "\nHV 只在同一问题签名下比较。SMS-EMOA 采用单后代更新及固定参考点；目标点数和决策点数分别记录。工厂结果是既有模型中的局部情景，不改变执行资格。\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def optimizer_software():
    import pymoo
    import scipy
    return {"pymoo": pymoo.__version__, "scipy": scipy.__version__, "native_threads": 1}
