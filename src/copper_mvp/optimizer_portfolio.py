"""Shared-budget optimizer portfolios with isolated algorithm state and point archives."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pymoo.core.population import Population
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from copper_mvp.common import WorkbenchError, digest, file_hash, utc_now, write_json
from copper_mvp.optimization import nondominated
from copper_mvp.optimizer_comparison import (
    TOLERANCE,
    EvaluationService,
    RegisteredProblem,
    make_optimizer,
    metric_configuration,
)

ARMS = ("NSGA-II", "SPEA2")
STRATEGIES = ("fixed_nsga2", "fixed_spea2", "equal_share", "adaptive")
POLICY = {
    "version": "optimizer-portfolio.g5d.v1",
    "population": 64,
    "batch": 64,
    "pilot_evaluations_per_arm": 128,
    "exploration_bonus": 0.0001,
    "retire_after_zero_gain_pulls": 4,
    "archive_exchange": True,
}


class PortfolioEvaluator(EvaluationService):
    def __init__(self, prepared, budget):
        super().__init__(prepared, budget)
        self.arm = None
        self.stage = None
        self.requests = 1
        self.archive_reuse = 0

    def evaluate(self, X, phase):
        self.requests += len(X)
        actual_phase = "pilot" if self.stage == "arm_pilot" and phase == "search" else phase
        result = super().evaluate(X, actual_phase)
        self.groups[-1].update(arm=self.arm, step=self.stage)
        return result

    def save(self, output):
        rows = []
        for group in self.groups:
            for X, F, G in zip(group["X"], group["F"], group["G"], strict=True):
                rows.append(
                    {
                        "phase": group["phase"],
                        "arm": group.get("arm"),
                        "step": group.get("step"),
                        **{f"x{i}": v for i, v in enumerate(X)},
                        **{f"f{i}": v for i, v in enumerate(F)},
                        **{f"g{i}": v for i, v in enumerate(G)},
                    }
                )
        pd.DataFrame(rows).to_csv(Path(output) / "evaluations.csv", index=False)


def evaluated_population(X, F, G):
    population = Population.new(
        X=np.asarray(X).copy(), F=np.asarray(F).copy(), G=np.asarray(G).copy() - TOLERANCE, H=np.empty((len(X), 0))
    )
    population.apply(lambda individual: individual.evaluated.update(("F", "G", "H")))
    return population


def archive_metrics(X, F, G, scale):
    feasible = np.flatnonzero(np.all(G <= TOLERANCE, axis=1))
    front = feasible[nondominated(F[feasible])] if len(feasible) else np.array([], dtype=int)
    normalized = F[front] / np.asarray(scale)
    inside = np.all(normalized <= 1.1, axis=1)
    hv = float(HV(ref_point=np.array([1.1, 1.1]))(normalized[inside])) if inside.any() else 0.0
    cv = np.maximum(G - TOLERANCE, 0).sum(axis=1)
    return {
        "front": front,
        "hv": hv,
        "feasible": len(feasible),
        "rate": len(feasible) / len(X),
        "min_violation": float(cv.min()) if len(cv) else None,
    }


def signed_archive(signature, X, F, G):
    value = {
        "problem_signature": signature,
        "X": np.asarray(X).tolist(),
        "F": np.asarray(F).tolist(),
        "G": np.asarray(G).tolist(),
    }
    return {**value, "content_hash": digest(value)}


def verify_archive(archive):
    if digest({k: v for k, v in archive.items() if k != "content_hash"}) != archive.get("content_hash"):
        raise WorkbenchError("前沿内容哈希不符", "OPTIMIZER_ARCHIVE_HASH")


def merge_archives(archives, signature):
    for archive in archives:
        verify_archive(archive)
    if any(a["problem_signature"] != signature for a in archives):
        raise WorkbenchError("不能合并不同问题签名的前沿", "OPTIMIZER_SIGNATURE")
    X = np.vstack([np.asarray(a["X"], float) for a in archives])
    F = np.vstack([np.asarray(a["F"], float) for a in archives])
    G = np.vstack([np.asarray(a["G"], float) for a in archives])
    _, indices = np.unique(X, axis=0, return_index=True)
    return X[np.sort(indices)], F[np.sort(indices)], G[np.sort(indices)]


def import_archive(archive, signature, evaluator):
    verify_archive(archive)
    X = np.asarray(archive["X"], float)
    prepared = evaluator.prepared
    if X.ndim != 2 or X.shape[1] != len(prepared.lower) or np.any(prepared.lower > X) or np.any(prepared.upper < X):
        raise WorkbenchError("热启动点与当前变量边界不匹配", "OPTIMIZER_WARM_START")
    if archive["problem_signature"] != signature:
        F, G, _ = evaluator.evaluate(X, "initial")
        return evaluated_population(X, F, G), {"re_evaluated": len(X), "reused": 0}
    F, G = np.asarray(archive["F"], float), np.asarray(archive["G"], float)
    if (
        F.shape != (len(X), 2)
        or G.shape != (len(X), prepared.constraints)
        or not np.isfinite(F).all()
        or not np.isfinite(G).all()
    ):
        raise WorkbenchError("热启动评估值不完整", "OPTIMIZER_WARM_START")
    evaluator.archive_reuse += len(X)
    return evaluated_population(X, F, G), {"re_evaluated": 0, "reused": len(X)}


class Arm:
    def __init__(self, name, evaluator, X, F, G, metric, seed):
        self.name = name
        self.algorithm = make_optimizer(
            name, evaluated_population(X, F, G), metric["scale"], len(evaluator.prepared.lower)
        )
        self.algorithm.setup(
            RegisteredProblem(evaluator), termination=("n_eval", evaluator.budget * 4), seed=seed, verbose=False
        )
        self.algorithm.next()
        self.X, self.F, self.G = X.copy(), F.copy(), G.copy()
        self.pulls = 0
        self.evaluations = 0
        self.gains = []
        self.rates = []
        self.stalled = False
        self.retired = False
        self.archive_points_received = 0

    def receive(self, archive, signature, evaluator):
        incoming, _ = import_archive(archive, signature, evaluator)
        merged = Population.merge(self.algorithm.pop, incoming)
        _, indices = np.unique(merged.get("X"), axis=0, return_index=True)
        merged = merged[np.sort(indices)]
        self.algorithm.pop = self.algorithm.survival.do(
            self.algorithm.problem,
            merged,
            n_survive=64,
            algorithm=self.algorithm,
            random_state=self.algorithm.random_state,
        )
        self.archive_points_received += len(incoming)

    def advance(self, evaluator, limit, stage):
        before = len(evaluator.groups)
        evaluator.arm = self.name
        evaluator.stage = stage
        self.algorithm.n_offsprings = min(64, limit)
        self.algorithm.next()
        groups = evaluator.groups[before:]
        if not groups:
            self.stalled = True
            return 0
        X = np.vstack([g["X"] for g in groups])
        F = np.vstack([g["F"] for g in groups])
        G = np.vstack([g["G"] for g in groups])
        self.X = np.vstack((self.X, X))
        self.F = np.vstack((self.F, F))
        self.G = np.vstack((self.G, G))
        self.pulls += 1
        self.evaluations += len(X)
        self.rates.append(float(np.all(G <= TOLERANCE, axis=1).mean()))
        return len(X)


def finalize(prepared, evaluator, metric, signature, spec, strategy, seed, reserve, stop, arms, trace, output, started):
    X, F, G = evaluator.search_points()
    measure = archive_metrics(X, F, G, metric["scale"])
    indices = measure["front"]
    decision_count = len(np.unique(X[indices], axis=0)) if len(indices) else 0
    if len(indices):
        _, unique = np.unique(np.round(F[indices], 10), axis=0, return_index=True)
        indices = indices[np.sort(unique)]
        indices = indices[np.argsort(F[indices, 0])]
    objective_count = len(indices)
    if len(indices) > reserve:
        indices = indices[np.unique(np.linspace(0, len(indices) - 1, reserve).astype(int))]
    verified_X = X[indices]
    evaluator.arm = None
    evaluator.stage = "independent_verification"
    if len(indices):
        verified_F, verified_G, details = evaluator.evaluate(verified_X, "verification")
        if (
            not np.allclose(verified_F, F[indices], rtol=1e-9, atol=1e-8)
            or not np.allclose(verified_G, G[indices], rtol=1e-9, atol=1e-8)
            or not np.all(verified_G <= TOLERANCE)
            or np.any(verified_X < prepared.lower - TOLERANCE)
            or np.any(verified_X > prepared.upper + TOLERANCE)
        ):
            raise WorkbenchError("独立前沿复算不一致", "CANDIDATE_AUDIT")
    else:
        verified_F = np.empty((0, 2))
        verified_G = np.empty((0, prepared.constraints))
        details = {}
    normalized = verified_F / np.asarray(metric["scale"])
    inside = np.all(normalized <= 1.1, axis=1)
    hv = float(HV(ref_point=np.array([1.1, 1.1]))(normalized[inside])) if inside.any() else 0.0
    igd = (
        float(IGDPlus(prepared.reference_front / np.asarray(metric["scale"]))(normalized))
        if prepared.reference_front is not None and len(normalized)
        else None
    )
    candidates = []
    for i, (x, f, g) in enumerate(zip(verified_X, verified_F, verified_G, strict=True)):
        candidate = {
            "id": "candidate_" + digest(x.tolist())[:16],
            "variables": dict(zip([v["name"] for v in spec["variables"]], x.tolist(), strict=True)),
            "f1": float(f[0]),
            "f2": float(f[1]),
            "constraints": g.tolist(),
            "audit_passed": True,
        }
        if prepared.context:
            currents = details["currents"][i]
            candidate.update(
                as_value=float(details["y"][i, 1]),
                all_currents={f"stage{s}_current_a": float(currents[j]) for j, s in enumerate(prepared.stages)},
                support_distance=float(
                    np.linalg.norm((prepared.historical_currents - currents) / prepared.current_scale, axis=1).min()
                )
                if len(prepared.historical_currents)
                else None,
            )
        candidates.append(candidate)
    archive = signed_archive(signature, verified_X, verified_F, verified_G)
    result = {
        "schema_version": "portfolio-result.g5d.v1",
        "problem_signature": signature,
        "strategy": strategy,
        "seed": seed,
        "status": "feasible" if candidates else "infeasible",
        "stop_reason": stop,
        "algorithm_executed": bool(len(prepared.lower)),
        "budget_mode": "total_equivalent_v2",
        "budget_limit": evaluator.budget,
        "evaluations": evaluator.counts,
        "total_evaluations": evaluator.used,
        "elementary_evaluations": evaluator.used,
        "logical_evaluation_requests": evaluator.requests,
        "archive_reused_points": evaluator.archive_reuse,
        "external_cache_hits": 0,
        "retries": 0,
        "metric": metric,
        "hv": hv,
        "igd_plus": igd,
        "igd_reference": "declared_analytic_front"
        if prepared.reference_front is not None
        else "no_independent_reference_front",
        "initial_population_hash": digest(X[:64].tolist()) if len(prepared.lower) else digest(X.tolist()),
        "feasible_rate": measure["rate"],
        "minimum_constraint_violation": measure["min_violation"],
        "decision_front_points": decision_count,
        "objective_front_points": objective_count,
        "verified_front_points": len(candidates),
        "outside_reference_box": int((~inside).sum()),
        "first_feasible_ms": evaluator.first_feasible_ms,
        "elapsed_ms": (perf_counter() - started) * 1000,
        "timing_scope": (
            "metric pilot, initialization, arm pilots, search, exchange and fin"
            "al verification; excludes problem assembly"
        ),
        "arms": {
            a.name: {
                "pulls": a.pulls,
                "evaluations": a.evaluations,
                "retired": a.retired,
                "stalled": a.stalled,
                "archive_points_received": a.archive_points_received,
            }
            for a in arms
        },
        "candidates": candidates,
        "warnings": prepared.warnings,
        "execution_authorized": False,
        "created_at": utc_now(),
    }
    evaluator.save(output)
    write_json(output / "archive.json", archive)
    write_json(output / "allocation_trace.json", trace)
    result["evidence_hashes"] = {
        name: file_hash(output / name)
        for name in ("evaluations.csv", "archive.json", "allocation_trace.json", "problem.json")
    }
    write_json(output / "result.json", result)
    return result


def run_portfolio(prepared, strategy, seed, budget, seconds, output, progress=lambda value: None):
    if strategy not in STRATEGIES:
        raise WorkbenchError("没有该组合策略", "OPTIMIZER_NOT_FOUND")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise WorkbenchError("该运行已有结果", "OPTIMIZER_RUN_EXISTS")
    started = perf_counter()
    evaluator = PortfolioEvaluator(prepared, budget)
    dimensions = len(prepared.lower)
    pilot_unit = (
        np.array(list(product((0.0, 0.5, 1.0), repeat=dimensions))).reshape(-1, dimensions)
        if dimensions
        else np.empty((1, 0))
    )
    pilot_X = prepared.lower + pilot_unit * (prepared.upper - prepared.lower)
    pilot_F, pilot_G, _ = evaluator.evaluate(pilot_X, "pilot")
    metric = metric_configuration(prepared, pilot_F)
    spec = {
        **prepared.specification(),
        "metric": metric,
        "budget_mode": "total_equivalent_v2",
        "total_budget": budget,
        "evaluator_version": "shared-portfolio-evaluator.g5d.v1",
        "evaluator_hashes": {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("optimization_problem.py", "optimizer_comparison.py", "optimizer_portfolio.py")
        },
        "external_cache_policy": "isolated per run; no cross-run cache",
        "reference_front_hash": digest(prepared.reference_front.tolist())
        if prepared.reference_front is not None
        else None,
    }
    signature = digest(spec)
    write_json(output / "problem.json", {"signature": signature, **spec})
    reserve = min(128, max(16, budget // 8))
    arms = []
    trace = []
    stop = "fixed_reference"
    if dimensions:
        if len(pilot_X) >= 64:
            raise WorkbenchError("当前组合仅支持可用初始种群内的低维问题", "OPTIMIZER_DIMENSION")
        initial_X = np.random.default_rng(seed).uniform(
            prepared.lower, prepared.upper, size=(64 - len(pilot_X), dimensions)
        )
        initial_F, initial_G, _ = evaluator.evaluate(initial_X, "initial")
        X, F, G = np.vstack((pilot_X, initial_X)), np.vstack((pilot_F, initial_F)), np.vstack((pilot_G, initial_G))
        names = ("NSGA-II",) if strategy == "fixed_nsga2" else ("SPEA2",) if strategy == "fixed_spea2" else ARMS
        if len(names) > 1 and budget - reserve - evaluator.used < 64 * len(names):
            names = ("NSGA-II",)
        arms = [Arm(name, evaluator, X, F, G, metric, seed) for name in names]
        pilot_per_arm = min(
            POLICY["pilot_evaluations_per_arm"], max(0, (budget - reserve - evaluator.used) // (2 * len(arms)))
        )
        pilot_per_arm = (pilot_per_arm // 64) * 64
        if len(arms) > 1 and pilot_per_arm == 0:
            pilot_per_arm = 64
        last_arm = None

        def pull(arm, stage):
            nonlocal last_arm
            before = archive_metrics(*evaluator.search_points(), metric["scale"])
            own_before = archive_metrics(arm.X, arm.F, arm.G, metric["scale"])
            reused = 0
            if (
                stage == "search"
                and len(arms) > 1
                and last_arm is not None
                and last_arm != arm.name
                and POLICY["archive_exchange"]
            ):
                all_X, all_F, all_G = evaluator.search_points()
                take = before["front"]
                if len(take) > 64:
                    take = take[np.unique(np.linspace(0, len(take) - 1, 64).astype(int))]
                if len(take):
                    archive = signed_archive(signature, all_X[take], all_F[take], all_G[take])
                    arm.receive(archive, signature, evaluator)
                    reused = len(take)
            left = budget - reserve - evaluator.used
            actual = arm.advance(evaluator, min(64, left), stage)
            after = archive_metrics(*evaluator.search_points(), metric["scale"])
            own_after = archive_metrics(arm.X, arm.F, arm.G, metric["scale"])
            gain = (
                max(0.0, own_after["hv"] - own_before["hv"])
                if stage == "arm_pilot"
                else max(0.0, after["hv"] - before["hv"])
            )
            arm.gains.append(gain / max(1, actual))
            trace.append(
                {
                    "step": len(trace),
                    "stage": stage,
                    "arm": arm.name,
                    "charged_evaluations": actual,
                    "total_evaluations": evaluator.used,
                    "hv_before": before["hv"],
                    "hv_after": after["hv"],
                    "gain_per_evaluation": gain / max(1, actual),
                    "credit_basis": "own_archive" if stage == "arm_pilot" else "shared_archive",
                    "feasible_rate": arm.rates[-1] if actual else None,
                    "archive_points_received": reused,
                    "minimum_violation": after["min_violation"],
                }
            )
            last_arm = arm.name
            return actual

        try:
            if len(arms) > 1:
                for _ in range(pilot_per_arm // 64):
                    for arm in arms:
                        if perf_counter() - started >= seconds:
                            break
                        pull(arm, "arm_pilot")
            stop = "total_budget"
            round_index = 0
            while evaluator.used < budget - reserve:
                if perf_counter() - started >= seconds:
                    stop = "time_budget"
                    break
                available = [a for a in arms if not a.stalled and not a.retired and a.algorithm.has_next()]
                if not available:
                    stop = "no_new_candidates"
                    break
                if strategy == "adaptive":
                    total = max(1, sum(a.pulls for a in arms))

                    def score(arm, total=total):
                        own = archive_metrics(arm.X, arm.F, arm.G, metric["scale"])
                        exploration = POLICY["exploration_bonus"] * math.sqrt(math.log(total + 1) / max(1, arm.pulls))
                        reward = float(np.mean(arm.gains[-4:])) if arm.gains else 0.0
                        return (
                            bool(own["feasible"]),
                            float(np.mean(arm.rates[-2:])) if arm.rates else 0.0,
                            reward + exploration if own["feasible"] else -own["min_violation"] + exploration,
                            -arm.pulls,
                        )

                    arm = max(available, key=score)
                else:
                    arm = available[round_index % len(available)]
                    round_index += 1
                actual = pull(arm, "search")
                if (
                    strategy == "adaptive"
                    and len(available) > 1
                    and len(arm.gains) >= POLICY["retire_after_zero_gain_pulls"]
                ):
                    patience = POLICY["retire_after_zero_gain_pulls"]
                    other_gain = max(
                        (np.mean(a.gains[-patience:]) for a in available if a is not arm and a.gains), default=0.0
                    )
                    if max(arm.gains[-patience:]) <= 1e-15 and other_gain > 1e-12:
                        arm.retired = True
                        trace[-1]["retirement_reason"] = "zero marginal gain while another arm is improving"
                if len(trace) % 4 == 0:
                    progress(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "evaluations": evaluator.used,
                            "remaining": budget - evaluator.used,
                        }
                    )
                if actual == 0 and all(a.stalled for a in arms):
                    stop = "no_new_candidates"
                    break
        except Exception as exc:
            evaluator.save(output)
            write_json(output / "allocation_trace.json", trace)
            write_json(
                output / "failure.json",
                {
                    "code": getattr(exc, "code", type(exc).__name__),
                    "charged_evaluations": evaluator.counts,
                    "total_evaluations": evaluator.used,
                    "message": str(exc)[:300],
                },
            )
            raise
    return finalize(
        prepared, evaluator, metric, signature, spec, strategy, seed, reserve, stop, arms, trace, output, started
    )
