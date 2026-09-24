from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from copper_mvp.bayesian_optimizers import ENTROPY_OPTIMIZERS
from copper_mvp.contracts import RunRequest
from copper_mvp.optimization_problem import build_problem
from copper_mvp.optimizer_comparison import compare_optimizers, run_optimizer
from copper_mvp.optimizer_registry import OPTIMIZERS, OptimizerComparisonRequest


def mathematical():
    return build_problem(None, None, RunRequest(task_type="optimize", mode="benchmark", request_key="test"))


def hv2d(F, reference):
    total = 0.0
    previous = reference[1]
    for x, y in F[np.argsort(F[:, 0])]:
        if x <= reference[0] and y < previous:
            total += (reference[0] - x) * (previous - y)
            previous = y
    return total


def test_three_optimizers_share_budget_inputs_reference_and_verified_math(tmp_path):
    request = OptimizerComparisonRequest(request_key="math", mode="benchmark", seeds=(17, 29), total_budget=128)
    result = compare_optimizers(None, None, request, tmp_path)
    assert result["runs"] == 6
    assert len({r["problem_signature"] for r in result["results"]}) == 1
    for seed in request.seeds:
        group = [r for r in result["results"] if r["seed"] == seed]
        assert len({r["initial_population_hash"] for r in group}) == 1
    for run in result["results"]:
        assert run["status"] == "feasible" and run["algorithm_executed"]
        assert sum(run["evaluations"].values()) == run["total_evaluations"] <= request.total_budget
        frame = pd.read_csv(tmp_path / "case_0" / str(run["seed"]) / run["optimizer_id"] / "evaluations.csv")
        assert len(frame) == run["total_evaluations"]
        for phase, count in run["evaluations"].items():
            assert int(frame.phase.eq(phase).sum()) == count
        X = frame[["x0", "x1"]].to_numpy()
        np.testing.assert_allclose(frame[["f0", "f1"]], np.column_stack(((X * X).sum(1), ((X - 2) ** 2).sum(1))))
        np.testing.assert_allclose(frame.g0, X.sum(1) - 3)
        front = np.array([[c["f1"], c["f2"]] for c in run["candidates"]])
        normalized = front / np.array(run["metric"]["scale"])
        assert np.isclose(run["hv"], hv2d(normalized, np.array([1.1, 1.1])))
        assert np.isfinite(run["igd_plus"])
        assert all(max(c["constraints"]) <= 1e-8 for c in run["candidates"])
        assert not run["execution_authorized"]


def test_infeasible_and_plateau_results_keep_their_meaning(tmp_path):
    for kind in ("infeasible", "plateau"):
        prepared = mathematical()

        def evaluate(X, current=kind):
            return np.ones((len(X), 2)), np.full((len(X), 1), 1.0 if current == "infeasible" else -1.0), {}

        prepared.evaluate = evaluate
        prepared.ref_F = np.ones((1, 2))
        prepared.ref_G = np.full((1, 1), 1.0 if kind == "infeasible" else -1.0)
        prepared.reference_front = None
        definition = prepared.specification()
        prepared.specification = lambda d=definition, k=kind: {**d, "test_fixture": k, "constraints": {"kind": k}}
        for optimizer in (name for name in OPTIMIZERS if name not in ENTROPY_OPTIMIZERS):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                run = run_optimizer(prepared, optimizer, 17, 128, 10, tmp_path / kind / optimizer)
            if kind == "infeasible":
                assert run["status"] == "infeasible" and run["candidates"] == [] and run["hv"] == 0
            else:
                assert run["status"] == "feasible"
                assert run["objective_front_points"] == 1
                assert run["decision_front_points"] > run["objective_front_points"]
            assert run["igd_plus"] is None
            assert run["total_evaluations"] <= 128


def test_spea2_run_state_does_not_leak_across_objective_ranges(tmp_path):
    first = run_optimizer(mathematical(), "SPEA2", 17, 128, 10, tmp_path / "first")
    shifted = mathematical()
    original = shifted.evaluate

    def evaluate(X):
        F, G, detail = original(X)
        return F + np.array([-20.0, 20.0]), G, detail

    shifted.evaluate = evaluate
    shifted.ref_F += np.array([-20.0, 20.0])
    shifted.reference_front = None
    definition = shifted.specification()
    shifted.specification = lambda: {**definition, "test_fixture": "shifted_objectives"}
    run_optimizer(shifted, "SPEA2", 29, 128, 10, tmp_path / "shifted")
    repeated = run_optimizer(mathematical(), "SPEA2", 17, 128, 10, tmp_path / "repeat")
    assert first["candidates"] == repeated["candidates"]
    assert first["hv"] == repeated["hv"]


def test_atomic_json_write_handles_a_brief_windows_reader(tmp_path):
    import os

    if os.name != "nt":
        return
    import ctypes
    import threading
    from ctypes import wintypes

    from copper_mvp.common import write_json

    path = tmp_path / "progress.json"
    write_json(path, {"step": 1})
    kernel = ctypes.windll.kernel32
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    release = threading.Timer(0.05, lambda: kernel.CloseHandle(handle))
    release.start()
    try:
        write_json(path, {"step": 2})
    finally:
        release.join()
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 2}
