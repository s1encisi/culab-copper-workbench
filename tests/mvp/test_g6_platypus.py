from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
from pymoo.core.population import Population

from copper_mvp.contracts import RunRequest
from copper_mvp.optimization_problem import build_problem
from copper_mvp.optimizer_comparison import EvaluationService, RegisteredProblem, compare_optimizers, run_optimizer
from copper_mvp.optimizer_registry import OptimizerComparisonRequest
from copper_mvp.platypus_methods import PLATYPUS_OPTIMIZERS, PlatypusAdapter, _native_components, load_platypus


def mathematical():
    return build_problem(None, None, RunRequest(task_type="optimize", mode="benchmark", request_key="platypus-test"))


@pytest.mark.parametrize("budget", [128, 257])
def test_platypus_methods_charge_clones_and_partial_batches_against_raw_ledger(tmp_path, budget):
    request = OptimizerComparisonRequest(
        request_key="platypus",
        mode="benchmark",
        seeds=(17,),
        total_budget=budget,
        optimizers=("NSGA-II",) + PLATYPUS_OPTIMIZERS,
    )
    comparison = compare_optimizers(None, None, request, tmp_path)
    assert len({r["problem_signature"] for r in comparison["results"]}) == 1
    assert len({r["initial_population_hash"] for r in comparison["results"]}) == 1
    for run in comparison["results"]:
        assert run["evaluations"]["search"] == budget - max(16, budget // 8) - 65
        assert run["status"] == "feasible" and run["stop_reason"] == "total_budget"
        frame = pd.read_csv(tmp_path / "case_0" / "17" / run["optimizer_id"] / "evaluations.csv")
        assert len(frame) == run["total_evaluations"] <= budget
        X = frame[["x0", "x1"]].to_numpy()
        assert np.all((X >= 0) & (X <= 3))
        np.testing.assert_allclose(frame[["f0", "f1"]], np.column_stack(((X * X).sum(1), ((X - 2) ** 2).sum(1))))
        np.testing.assert_allclose(frame.g0, X.sum(1) - 3)
        assert all(max(c["constraints"]) <= 1e-8 for c in run["candidates"])


def test_platypus_randomness_does_not_change_python_global_state_or_neighboring_runs(tmp_path):
    before = random.getstate()
    first = run_optimizer(mathematical(), "PAES", 17, 257, 120, tmp_path / "first")
    run_optimizer(mathematical(), "IBEA", 29, 257, 120, tmp_path / "interleaved")
    repeated = run_optimizer(mathematical(), "PAES", 17, 257, 120, tmp_path / "repeat")
    assert random.getstate() == before
    assert first["candidates"] == repeated["candidates"]


def test_ibea_mating_and_removal_use_constraint_first_comparison():
    p = load_platypus()
    comparator_type, _, _, _ = _native_components(p)
    problem = p.Problem(1, 2, 1)
    problem.types[:] = p.Real(0, 1)
    problem.constraints[:] = "<=0"
    population = []
    for violation, fitness in ((0, 1e12), (1, -1e12), (2, -1e15)):
        solution = p.Solution(problem)
        solution.constraint_violation, solution.fitness = violation, fitness
        population.append(solution)
    comparator = comparator_type()
    algorithm = p.IBEA(problem, fitness_comparator=comparator)
    algorithm.population = population
    assert comparator.compare(population[0], population[1]) < 0
    assert algorithm._find_worst() == 2


def test_multiobjective_cma_distribution_responds_to_second_objective():
    means = []
    for reverse in (False, True):
        prepared = mathematical()
        u = np.linspace(0, 1, 64)
        X = 3 * np.column_stack((u, 1 - u))
        F = np.column_stack((np.zeros(64), 1 - u if reverse else u))
        population = Population.new(X=X, F=F, G=-np.ones((64, 1)))
        evaluator = EvaluationService(prepared, 257)
        algorithm = PlatypusAdapter("MO-CMA-ES", population, [1, 3])
        algorithm.setup(RegisteredProblem(evaluator), seed=17)
        algorithm.next()
        means.append(algorithm.native.xmean[0])
        assert abs(algorithm.native.C[1][0]) > 1e-4
    assert means[0] < 0.5 < means[1]
