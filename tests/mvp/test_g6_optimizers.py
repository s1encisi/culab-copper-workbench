from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem

from copper_mvp.optimizer_comparison import compare_optimizers
from copper_mvp.optimizer_methods import EXTENDED_OPTIMIZERS, BudgetedGDE3, make_extended_optimizer
from copper_mvp.optimizer_registry import LEGACY_OPTIMIZERS, OptimizerComparisonRequest


@pytest.mark.parametrize("budget", [128, 257])
def test_extended_algorithms_share_raw_problem_initial_points_and_total_budget(tmp_path, budget):
    request = OptimizerComparisonRequest(request_key="extended", mode="benchmark", seeds=(17,),
        total_budget=budget, seconds_per_run=120, optimizers=("NSGA-II",) + EXTENDED_OPTIMIZERS)
    result = compare_optimizers(None, None, request, tmp_path)
    assert OptimizerComparisonRequest(request_key="default", mode="benchmark").optimizers == LEGACY_OPTIMIZERS
    assert len({r["initial_population_hash"] for r in result["results"]}) == 1
    assert len({r["problem_signature"] for r in result["results"]}) == 1
    for run in result["results"]:
        assert run["status"] == "feasible" and run["algorithm_executed"]
        frame = pd.read_csv(tmp_path / "case_0" / "17" / run["optimizer_id"] / "evaluations.csv")
        assert len(frame) == run["total_evaluations"] <= budget
        assert run["evaluations"]["search"] == budget - max(16, budget // 8) - 65
        X = frame[["x0", "x1"]].to_numpy()
        assert np.all((X >= 0) & (X <= 3))
        np.testing.assert_allclose(frame[["f0", "f1"]], np.column_stack(((X * X).sum(1), ((X - 2)**2).sum(1))))
        np.testing.assert_allclose(frame.g0, X.sum(1) - 3)
        front = np.array([[c["f1"], c["f2"]] for c in run["candidates"]]) / np.array(run["metric"]["scale"])
        previous, hv = 1.1, 0.0
        for x, y in front[np.argsort(front[:, 0])]:
            if x <= 1.1 and y < previous:
                hv += (1.1 - x) * (previous - y)
                previous = y
        assert np.isclose(hv, run["hv"]) and np.isfinite(run["igd_plus"])
        assert all(max(c["constraints"]) <= 1e-8 for c in run["candidates"])


def test_moead_compares_constraints_before_extreme_objective_values():
    population = Population.new(X=np.zeros((64, 2)), F=np.ones((64, 2)), G=np.full((64, 1), -1.0))
    algorithm = make_extended_optimizer("MOEA-D", population, 2)
    problem = Problem(n_var=2, n_obj=2, n_ieq_constr=1, xl=0, xu=1)
    algorithm.setup(problem, termination=("n_eval", 128), seed=17)
    algorithm.pop = population
    algorithm._initialize_advance(population)
    infeasible = Population.new(X=np.array([[0.1, 0.1]]), F=np.array([[-1e9, -1e9]]), G=np.array([[1.0]]))[0]
    algorithm._replace(0, infeasible)
    np.testing.assert_array_equal(algorithm.pop.get("F"), np.ones((64, 2)))
    np.testing.assert_array_equal(algorithm.ideal, [1.0, 1.0])
    algorithm.pop = Population.new(X=np.zeros((64, 2)), F=np.zeros((64, 2)), G=np.ones((64, 1)))
    feasible = Population.new(X=np.array([[0.2, 0.2]]), F=np.array([[100.0, 100.0]]), G=np.array([[-1.0]]))[0]
    algorithm._replace(0, feasible)
    assert all(float(algorithm.pop[i].CV[0]) == 0 for i in algorithm.neighbors[0])


def test_gde3_partial_batch_keeps_unvisited_parents_and_rejects_infeasible_trial():
    algorithm = BudgetedGDE3(pop_size=4)
    problem = Problem(n_var=1, n_obj=2, n_ieq_constr=1, xl=0, xu=10)
    algorithm.setup(problem, termination=("n_eval", 128), seed=17)
    algorithm.pop = Population.new(X=np.array([[0], [1], [2], [3]]), F=np.array([[0, 3], [1, 2], [2, 1], [3, 0]], dtype=float), G=-np.ones((4, 1)))
    algorithm.trial_parent_indices = np.array([1, 3])
    trial = Population.new(X=np.array([[7], [8]]), F=np.array([[-100, -100], [2.5, -0.1]]), G=np.array([[1], [-1]]))
    algorithm._advance(trial)
    assert set(algorithm.pop.get("X")[:, 0]) == {0, 1, 2, 8}

@pytest.mark.parametrize("name", ["C-TAEA", "MOEA-D"])
def test_narrow_feasible_region_preserves_only_verified_candidates(tmp_path, name):
    from copper_mvp.contracts import RunRequest
    from copper_mvp.optimization_problem import build_problem
    from copper_mvp.optimizer_comparison import run_optimizer
    prepared = build_problem(None, None, RunRequest(task_type="optimize", mode="benchmark", request_key="thin"))
    original = prepared.evaluate
    def evaluate(X):
        F, _, details = original(X)
        return F, (np.abs(X.sum(axis=1) - 3) - 1e-7)[:, None], details
    prepared.evaluate = evaluate
    prepared.ref_F, prepared.ref_G, _ = evaluate(prepared.reference_x)
    prepared.reference_front = None
    definition = prepared.specification()
    prepared.specification = lambda: {**definition, "test_fixture": "abs(x0+x1-3)<=1e-7"}
    run = run_optimizer(prepared, name, 29, 257, 120, tmp_path)
    assert run["status"] == "feasible" and run["evaluations"]["search"] > 0
    assert run["igd_plus"] is None
    assert all(abs(sum(c["variables"].values()) - 3) <= 1.1e-7 for c in run["candidates"])

