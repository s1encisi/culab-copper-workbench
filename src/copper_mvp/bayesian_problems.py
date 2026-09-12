"""Additional mathematical problem for explicitly unconstrained entropy search."""
import numpy as np
from copper_mvp.optimization_problem import build_problem


def build_comparison_problem(data, models, request, event_id, seed):
    prepared = build_problem(data, models, request.problem_request(event_id, seed))
    if request.mode == "benchmark" and request.benchmark_problem == "unconstrained_quadratic":
        def evaluate(X):
            F = np.column_stack(((X*X).sum(axis=1), ((X-2)**2).sum(axis=1)))
            return F, np.empty((len(X), 0)), {"variables": X}
        prepared.evaluate = evaluate
        prepared.constraints = 0
        prepared.ref_G = np.empty((1, 0))
        t = np.linspace(0, 2, 1001)
        prepared.reference_front = np.column_stack((2*t*t, 2*(t-2)**2))
        base_specification = prepared.specification
        def specification():
            record = base_specification()
            record["benchmark_problem"] = "unconstrained_quadratic"
            record["constraints"] = {"kind": "unconstrained", "normalized_tolerance": 1e-8}
            return record
        prepared.specification = specification
    return prepared
