"""Fixed-budget scalarization sweeps with jointly evaluated objectives/constraints."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy

from copper_mvp.common import WorkbenchError, safe

SCALAR_OPTIMIZERS = ("Epsilon-Constraint", "Augmented-Chebyshev", "Weighted-Sum", "NBI", "NNC")
TOLERANCE = 1e-8
GEOMETRY_TOLERANCE = 1e-6
SPAN_TOLERANCE = 1e-10


def scalarization_spec(name):
    designs = {
        "Epsilon-Constraint": "O17",
        "Augmented-Chebyshev": "O18",
        "Weighted-Sum": "O19",
        "NBI": "O20",
        "NNC": "O21",
    }
    equations = {
        "Epsilon-Constraint": "min y2 subject to y1 <= lambda",
        "Augmented-Chebyshev": "min max(w*y) + 0.001*sum(w*y); w=(1-lambda,lambda)",
        "Weighted-Sum": "min (1-lambda)*y1 + lambda*y2",
        "NBI": "max t subject to y=(lambda,1-lambda)+t*(-1,-1)/sqrt(2)",
        "NNC": "min y2 subject to (1,-1) dot (y-(lambda,1-lambda)) <= 0",
    }
    return {
        "optimizer_id": name,
        "design_id": designs[name],
        "method_version": "g6e.scalar.v3",
        "mechanism": equations[name],
        "package": "scipy",
        "package_version": "1.17.1",
        "inner_solver": "PRIMA COBYLA combined calcfc",
        "variables": "continuous",
        "objectives": 2,
        "inequality_constraints": True,
        "constraint_handling": "explicit_physical_and_preference_constraints",
        "parameters": {
            "anchor_fraction": 0.2,
            "anchor_targets": 2,
            "starts_per_subproblem": 2,
            "interior_preferences_max": 9,
            "preference_grid": "equally_spaced_inside_(0,1)",
            "normalization": "frozen_best_feasible_anchor_ideal_and_payoff_spans",
            "rhobeg": 0.2,
            "rhoend": 1e-6,
            "physical_constraint_tolerance": TOLERANCE,
            "nbi_equality_tolerance": GEOMETRY_TOLERANCE,
            "anchor_span_tolerance": SPAN_TOLERANCE,
            "decision_scaling": "unit_box",
            "out_of_box_probe_evaluation": "clipped_to_declared_box_with_explicit_solver_bounds",
            "budget_counting": "joint_calcfc_calls_plus_one_charged_return_check_per_subproblem; no_callback_cache",
            "convergence": "recorded_per_subproblem_separately_from_candidate_feasibility",
        },
        "plant_model_profiles": ["DeltaRidge"] if name == "NBI" else ["DeltaHGB", "DeltaRidge"],
        "smooth_proxy_preferred": name in ("NBI", "NNC"),
        "status": "registered",
        "execution_authorized": False,
    }


def scalar_value_constraints(name, y, parameter, auxiliary=0.0):
    """y is measured relative to the frozen anchor ideal and positive scales."""
    if name == "Epsilon-Constraint":
        return float(y[1]), np.array([y[0] - parameter])
    if name == "Weighted-Sum":
        return float(np.dot([1 - parameter, parameter], y)), np.empty(0)
    if name == "Augmented-Chebyshev":
        weighted = np.array([1 - parameter, parameter]) * y
        return float(weighted.max() + 0.001 * weighted.sum()), np.empty(0)
    base = np.array([parameter, 1 - parameter])
    if name == "NNC":
        return float(y[1]), np.array([np.dot([1.0, -1.0], y - base)])
    if name == "NBI":
        residual = y - base + auxiliary / np.sqrt(2)
        return -float(auxiliary), np.r_[residual - GEOMETRY_TOLERANCE, -residual - GEOMETRY_TOLERANCE]
    raise ValueError("Unknown scalarization")


class _StopSweep(Exception):
    pass


class ScalarizationAdapter:
    def __init__(self, name, population, scale):
        self.name, self.initial, self.metric_scale = name, population, np.asarray(scale)
        self.deadline = float("inf")
        self.n_offsprings = 64
        self.initialized = False
        self.stop_reason = None
        self.trace = []
        self.anchor_record = None
        self.preference_jobs = None
        self.preference_index = 0
        self.ideal = np.zeros(2)
        self.normalization = np.ones(2)

    def setup(self, problem, termination=None, seed=None, verbose=False):
        if scipy.__version__ != "1.17.1":
            raise WorkbenchError("标量化联合回调需要 SciPy 1.17.1", "OPTIMIZER_DEPENDENCY")
        self.evaluator = problem.evaluator
        prepared = self.evaluator.prepared
        if self.name == "NBI" and prepared.context and prepared.request.model_profile != "DeltaRidge":
            raise WorkbenchError("NBI 当前仅在 DeltaRidge 历史代理上启用", "OPTIMIZER_CAPABILITY")
        self.lower, self.span = problem.xl.copy(), problem.xu - problem.xl
        self.n_var, self.n_constraints = problem.n_var, problem.n_ieq_constr
        self.rng = np.random.default_rng(seed)
        self.reserve = min(128, max(16, self.evaluator.budget // 8))
        self.limit = self.evaluator.budget - self.reserve
        self.minimum_calls = self.n_var + 3
        self.preference_minimum_calls = self.n_var + 3 + int(self.name == "NBI")
        available = self.limit - self.evaluator.used
        if available < 4 * self.minimum_calls + 4 * self.preference_minimum_calls:
            raise WorkbenchError("预算不足以完成重复锚点和至少两个偏好子问题", "SCALARIZATION_BUDGET")
        anchor_budget = max(4 * self.minimum_calls, int(0.2 * available))
        anchor_budget = min(anchor_budget, available - 4 * self.preference_minimum_calls)
        self.anchor_quotas = [anchor_budget // 4 + int(i < anchor_budget % 4) for i in range(4)]
        self.anchor_jobs = [(target, restart) for target in (0, 1) for restart in (0, 1)]
        self.initial_x = self.initial.get("X").copy()

    def has_next(self):
        return self.stop_reason is None

    def _seen(self):
        X, F, G = self.evaluator.search_points()
        return X, F / self.metric_scale, G

    def _coordinates(self, X, F, phase, parameter):
        unit = (X - self.lower) / self.span
        if phase == "preference" and self.name == "NBI":
            y = (F - self.ideal) / self.normalization
            auxiliary = -float(np.sum(y - np.array([parameter, 1 - parameter]))) / np.sqrt(2)
            return np.r_[unit, auxiliary]
        return unit

    def _values(self, F, coordinates, phase, parameter):
        if phase == "anchor":
            return float(F[int(parameter)]), np.empty(0)
        y = (F - self.ideal) / self.normalization
        return scalar_value_constraints(self.name, y, parameter, coordinates[-1] if self.name == "NBI" else 0)

    def _start(self, phase, parameter, restart):
        X, F, G = self._seen()
        if restart:
            index = int(self.rng.integers(len(self.initial_x)))
            # The first 64 ledger points are exactly the common initial samples.
            return self._coordinates(X[index], F[index], phase, parameter)
        scores, violations, points = [], [], []
        for x, f, g in zip(X, F, G, strict=True):
            coordinates = self._coordinates(x, f, phase, parameter)
            score, extra = self._values(f, coordinates, phase, parameter)
            scores.append(score)
            violations.append(float(np.maximum(np.r_[g, extra], 0).sum()))
            points.append(coordinates)
        return points[int(np.lexsort((scores, violations))[0])]

    def _finish_anchors(self):
        X, F, G = self._seen()
        feasible = np.flatnonzero(np.all(G <= TOLERANCE, axis=1))
        if not len(feasible):
            self.stop_reason = "no_feasible_anchor"
            return
        indices = [feasible[np.lexsort((F[feasible, 1 - target], F[feasible, target]))[0]] for target in (0, 1)]
        anchors = F[indices]
        self.ideal = np.array([anchors[0, 0], anchors[1, 1]])
        span = np.array([anchors[1, 0] - self.ideal[0], anchors[0, 1] - self.ideal[1]])
        self.normalization = np.where(span > SPAN_TOLERANCE, span, 1.0)
        self.anchor_record = {
            "variables": X[indices].tolist(),
            "raw_objectives": (anchors * self.metric_scale).tolist(),
            "scaled_ideal": self.ideal.tolist(),
            "payoff_spans": span.tolist(),
            "normalization": self.normalization.tolist(),
            "source": "best_feasible_observed_points_after_all_four_anchor_searches",
        }
        if self.name in ("NBI", "NNC") and np.any(span <= SPAN_TOLERANCE):
            self.stop_reason = "degenerate_anchor_geometry"
            return
        remaining = self.limit - self.evaluator.used
        count = min(9, remaining // (2 * self.preference_minimum_calls))
        if count < 2:
            self.stop_reason = "insufficient_remaining_sweep_budget"
            return
        parameters = np.linspace(0, 1, count + 2)[1:-1]
        jobs = [(float(parameter), restart) for parameter in parameters for restart in (0, 1)]
        self.preference_jobs = [
            (parameter, restart, remaining // len(jobs) + int(i < remaining % len(jobs)))
            for i, (parameter, restart) in enumerate(jobs)
        ]

    def _solve(self, phase, parameter, restart, quota):
        from scipy._lib.pyprima.cobyla.cobyla import cobyla
        from scipy._lib.pyprima.common.message import get_info_string

        x0 = self._start(phase, parameter, restart)
        extra_count = (
            4
            if phase == "preference" and self.name == "NBI"
            else int(phase == "preference" and self.name in ("NNC", "Epsilon-Constraint"))
        )
        lower, upper = np.zeros(len(x0)), np.ones(len(x0))
        if len(x0) > self.n_var:
            lower[-1], upper[-1] = -np.inf, np.inf
        before = self.evaluator.used
        last_record = None
        clipped = 0

        def calcfc(coordinates):
            nonlocal clipped, last_record
            if perf_counter() >= self.deadline:
                self.stop_reason = "time_budget"
                raise _StopSweep()
            if self.evaluator.used >= self.limit or self.evaluator.used - before >= quota:
                self.stop_reason = "total_budget"
                raise _StopSweep()
            unit = np.clip(coordinates[: self.n_var], 0, 1)
            clipped += int(np.any(unit != coordinates[: self.n_var]))
            physical = self.lower + unit * self.span
            F, G, _ = self.evaluator.evaluate(physical[None, :], "search")
            value, extra = self._values(F[0] / self.metric_scale, coordinates, phase, parameter)
            constraints = np.r_[G[0], extra]
            last_record = {
                "variables": physical.tolist(),
                "objectives": F[0].tolist(),
                "constraints": G[0].tolist(),
                "preference_constraints": extra.tolist(),
                "scalar_objective": value,
                "solver_coordinates": coordinates.tolist(),
            }
            return value, constraints

        summary = {
            "stage": phase,
            "parameter": parameter,
            "restart": restart,
            "quota": quota,
            "ledger_start": before,
            "start_coordinates": x0.tolist(),
        }
        try:
            result = cobyla(
                calcfc,
                self.n_constraints + extra_count,
                x0,
                xl=lower,
                xu=upper,
                rhobeg=0.2,
                rhoend=1e-6,
                ctol=TOLERANCE,
                maxfun=quota - 1,
                maxhist=quota,
                iprint=0,
            )
            returned_value, returned_constraints = calcfc(result.x)
            unit = result.x[: self.n_var]
            maxcv = float(np.r_[0.0, returned_constraints, -unit, unit - 1].max())
            difference = float(abs(returned_value - result.f))
            consistent = difference <= 1e-8 * max(1.0, abs(result.f))
            summary.update(
                solver_info=int(result.info),
                solver_reason=get_info_string("COBYLA", result.info),
                solver_nfev=int(result.nf),
                return_verification_evaluations=1,
                subproblem_feasible=bool(maxcv <= TOLERANCE),
                solver_reported_maxcv=float(result.cstrv),
                solver_return_value_difference=difference,
                solver_return_consistent=consistent,
                solver_converged=bool(maxcv <= TOLERANCE and consistent and result.info in (0, 1)),
                maxcv=maxcv,
                returned=last_record,
            )
            if result.nf + 1 != self.evaluator.used - before:
                raise WorkbenchError("内层求值次数与原始账本不一致", "SCALARIZATION_ACCOUNTING")
        except _StopSweep:
            summary.update(solver_reason=self.stop_reason, solver_converged=False, subproblem_feasible=None)
        summary.update(
            actual_evaluations=self.evaluator.used - before, ledger_stop=self.evaluator.used, clipped_probes=clipped
        )
        self.trace.append(safe(summary))

    def next(self):
        if not self.initialized:
            self.initialized = True
            return
        if len(self.trace) < 4:
            i = len(self.trace)
            target, restart = self.anchor_jobs[i]
            self._solve("anchor", target, restart, self.anchor_quotas[i])
            return
        if self.preference_jobs is None:
            self._finish_anchors()
            if self.stop_reason is not None:
                return
        parameter, restart, quota = self.preference_jobs[self.preference_index]
        self._solve("preference", parameter, restart, quota)
        self.preference_index += 1
        if self.preference_index == len(self.preference_jobs) and self.stop_reason is None:
            self.stop_reason = "scalarization_sweep_complete"

    def details(self):
        return safe(
            {
                "status": self.stop_reason,
                "anchors": self.anchor_record,
                "jobs": self.trace,
                "anchor_evaluations": sum(r["actual_evaluations"] for r in self.trace if r["stage"] == "anchor"),
                "preference_evaluations": sum(
                    r["actual_evaluations"] for r in self.trace if r["stage"] == "preference"
                ),
                "converged_subproblems": sum(r["solver_converged"] for r in self.trace),
                "physically_verified_front_is_reported_separately": True,
            }
        )
