from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from copper_mvp.common import WorkbenchError
from copper_mvp.contracts import RunRequest
from copper_mvp.optimization_problem import build_problem
from copper_mvp.optimizer_comparison import compare_optimizers, run_optimizer
from copper_mvp.optimizer_registry import OptimizerComparisonRequest
from copper_mvp.scalarization import SCALAR_OPTIMIZERS, scalar_value_constraints


def mathematical():
    return build_problem(None, None, RunRequest(task_type="optimize", mode="benchmark", request_key="scalar-test"))


@pytest.mark.parametrize("budget", [128, 384])
def test_scalar_sweeps_charge_anchors_returns_and_every_joint_call(tmp_path, budget):
    request = OptimizerComparisonRequest(
        request_key="sweep",
        mode="benchmark",
        seeds=(17,),
        total_budget=budget,
        optimizers=("NSGA-II",) + SCALAR_OPTIMIZERS,
    )
    comparison = compare_optimizers(None, None, request, tmp_path)
    assert len({r["initial_population_hash"] for r in comparison["results"]}) == 1
    assert len({r["problem_signature"] for r in comparison["results"]}) == 1
    for run in comparison["results"]:
        frame = pd.read_csv(tmp_path / "case_0" / "17" / run["optimizer_id"] / "evaluations.csv")
        assert len(frame) == run["total_evaluations"] <= budget
        X = frame[["x0", "x1"]].to_numpy()
        assert np.all((X >= 0) & (X <= 3))
        np.testing.assert_allclose(frame[["f0", "f1"]], np.column_stack(((X * X).sum(1), ((X - 2) ** 2).sum(1))))
        np.testing.assert_allclose(frame.g0, X.sum(1) - 3, atol=1e-12)
        if run["optimizer_id"] == "NSGA-II":
            continue
        detail = run["scalarization"]
        assert detail["status"] == "scalarization_sweep_complete"
        jobs = detail["jobs"]
        assert len([r for r in jobs if r["stage"] == "anchor"]) == 4
        assert len({r["parameter"] for r in jobs if r["stage"] == "preference"}) >= 2
        assert sum(r["actual_evaluations"] for r in jobs) == run["evaluations"]["search"]
        for job in jobs:
            assert (
                job["actual_evaluations"] == job["solver_nfev"] + job["return_verification_evaluations"] <= job["quota"]
            )
            assert job["return_verification_evaluations"] == 1
            assert job["ledger_stop"] - job["ledger_start"] == job["actual_evaluations"]


@pytest.mark.parametrize("name", ["NBI", "NNC"])
def test_normal_methods_solve_the_known_middle_front_intersection(tmp_path, name):
    run = run_optimizer(mathematical(), name, 17, 2048, 120, tmp_path)
    middle = [r for r in run["scalarization"]["jobs"] if r["stage"] == "preference" and np.isclose(r["parameter"], 0.5)]
    feasible = [r for r in middle if r["subproblem_feasible"]]
    assert feasible
    # On the analytic Pareto curve x1=x2=t, equal normalized objectives give this root.
    expected = (-24 + np.sqrt(936)) / 8
    best = min(feasible, key=lambda r: np.linalg.norm(np.array(r["returned"]["variables"]) - expected))
    np.testing.assert_allclose(best["returned"]["variables"], [expected, expected], atol=0.02)
    assert max(best["returned"]["preference_constraints"]) <= 1e-8
    assert len(best["returned"]["solver_coordinates"]) == (3 if name == "NBI" else 2)


def test_scalar_functions_and_normal_halfspace_have_distinct_geometry():
    value, constraints = scalar_value_constraints("Weighted-Sum", np.array([0.2, 0.8]), 0.25)
    assert np.isclose(value, 0.35) and len(constraints) == 0
    value, constraints = scalar_value_constraints("Augmented-Chebyshev", np.array([0.2, 0.8]), 0.25)
    assert np.isclose(value, 0.20035) and len(constraints) == 0
    _, constraints = scalar_value_constraints("Epsilon-Constraint", np.array([0.2, 0.8]), 0.1)
    np.testing.assert_allclose(constraints, [0.1])
    value, constraints = scalar_value_constraints("NBI", np.array([0.1, 0.5]), 0.3, 0.2 * np.sqrt(2))
    np.testing.assert_allclose(constraints, [-1e-6] * 4, atol=1e-15)
    _, constraints = scalar_value_constraints("NNC", np.array([0.15, 0.4]), 0.3)
    np.testing.assert_allclose(constraints, [0.15])


def test_nbi_hgb_capability_is_rejected_before_a_comparison():
    with pytest.raises(ValueError, match="DeltaRidge"):
        OptimizerComparisonRequest(request_key="capability", mode="plant", event_ids=("event",), optimizers=("NBI",))


def test_failed_evaluation_keeps_charged_budget_and_completed_values(tmp_path):
    prepared = mathematical()
    original = prepared.evaluate
    measured = {"evaluations": 1, "calls": 0}

    def fail_after_computing(X):
        result = original(X)
        measured["evaluations"] += len(X)
        measured["calls"] += 1
        if measured["calls"] == 2:
            raise WorkbenchError("synthetic post-compute failure", "MODEL_FAILURE")
        return result

    prepared.evaluate = fail_after_computing
    with pytest.raises(WorkbenchError, match="synthetic"):
        run_optimizer(prepared, "NSGA-II", 17, 128, 120, tmp_path)
    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["total_evaluations"] == measured["evaluations"] == 65
    assert failure["completed_evaluation_rows"] == len(pd.read_csv(tmp_path / "evaluations.csv")) == 10
    assert failure["charged_without_saved_values"] == 55
    assert not (tmp_path / "result.json").exists()


def test_augmented_chebyshev_returns_an_unsupported_concave_front_point(tmp_path):
    def problem():
        prepared = mathematical()
        prepared.lower, prepared.upper = np.array([0.0]), np.array([1.0])
        prepared.reference_x = np.array([[0.5]])

        def evaluate(X):
            return np.column_stack((X[:, 0], 1 - X[:, 0] ** 2)), -np.ones((len(X), 1)), {}

        prepared.evaluate = evaluate
        prepared.ref_F, prepared.ref_G, _ = evaluate(prepared.reference_x)
        u = np.linspace(0, 1, 1001)
        prepared.reference_front = np.column_stack((u, 1 - u**2))
        definition = prepared.specification()
        prepared.specification = lambda: {**definition, "test_fixture": "concave_front_x_and_1_minus_x_squared"}
        return prepared

    weighted = run_optimizer(problem(), "Weighted-Sum", 17, 2048, 120, tmp_path / "weighted")
    chebyshev = run_optimizer(problem(), "Augmented-Chebyshev", 17, 2048, 120, tmp_path / "chebyshev")

    def middle(run):
        return [
            r["returned"]["variables"][0]
            for r in run["scalarization"]["jobs"]
            if r["stage"] == "preference" and np.isclose(r["parameter"], 0.5)
        ]

    # Equal weighted sums minimize this concave scalar function at the endpoints.
    assert all(min(abs(x), abs(x - 1)) < 0.005 for x in middle(weighted))
    expected = (np.sqrt(5) - 1) / 2
    assert any(abs(x - expected) < 0.005 for x in middle(chebyshev))
