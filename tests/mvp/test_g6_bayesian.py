from __future__ import annotations

from time import perf_counter
import numpy as np
import pytest
from pymoo.core.population import Population

from copper_mvp.bayesian_optimizers import BAYESIAN_OPTIMIZERS, BayesianAdapter
from copper_mvp.bayesian_problems import build_comparison_problem
from copper_mvp.common import file_hash
from copper_mvp.neural_runtime import load_tensor_runtime, tensor_random_scope
from copper_mvp.optimizer_comparison import EvaluationService, RegisteredProblem
from copper_mvp.optimizer_registry import OptimizerComparisonRequest


def test_entropy_methods_require_explicit_unconstrained_problem():
    for method in ("MES", "JES"):
        with pytest.raises(ValueError, match="无约束"):
            OptimizerComparisonRequest(request_key="cap", mode="plant", event_ids=("event",), optimizers=(method,))
        with pytest.raises(ValueError, match="无约束"):
            OptimizerComparisonRequest(request_key="cap", mode="benchmark", optimizers=(method,))
    request = OptimizerComparisonRequest(request_key="entropy", mode="benchmark", benchmark_problem="unconstrained_quadratic", optimizers=("MES",))
    prepared = build_comparison_problem(None, None, request, None, 17)
    F, G, _ = prepared.evaluate(np.array([[2.0, 2.0]]))
    np.testing.assert_allclose(F, [[8.0, 0.0]])
    assert G.shape == (1, 0) and prepared.constraints == 0
    np.testing.assert_allclose(prepared.reference_front[-1], [8.0, 0.0])


@pytest.mark.parametrize("name", BAYESIAN_OPTIMIZERS)
def test_one_bayesian_iteration_fits_gp_proposes_and_evaluates_real_candidates(tmp_path, name):
    torch = load_tensor_runtime()
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    problem_id = "unconstrained_quadratic" if name in ("MES", "JES") else "constrained_quadratic"
    request = OptimizerComparisonRequest(request_key="iteration", mode="benchmark", benchmark_problem=problem_id, optimizers=(name,))
    prepared = build_comparison_problem(None, None, request, None, 17)
    evaluator = EvaluationService(prepared, 128)
    X = np.random.default_rng(17).uniform(prepared.lower, prepared.upper, (64, 2))
    F, G, _ = evaluator.evaluate(X, "initial")
    scale = np.array([18., 8.])
    population = Population.new(X=X, F=F/scale, G=G-1e-8)
    adapter = BayesianAdapter(name, population, scale)
    adapter.setup(RegisteredProblem(evaluator, scale), seed=17)
    adapter.artifact_dir = tmp_path
    adapter.deadline = perf_counter()+120
    adapter.next()
    rng_before = torch.random.get_rng_state().clone()
    adapter.n_offsprings = 2
    adapter.next()
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert evaluator.used == 67 and evaluator.counts["search"] == 2
    trace = adapter.trace[0]
    assert trace["status"] == "evaluated" and trace["gp_fit_steps"] > 0
    assert trace["acquisition_calls"] > 0 and trace["modeled_constraint_outputs"] == prepared.constraints
    expected_class = {"ParEGO": "qLogNParEGO", "NEHVI": "qLogNoisyExpectedHypervolumeImprovement",
        "MES": "qLowerBoundMultiObjectiveMaxValueEntropySearch", "JES": "qLowerBoundMultiObjectiveJointEntropySearch"}[name]
    assert trace["acquisition_class"] == expected_class
    artifact = tmp_path/trace["gp_artifact"]
    assert file_hash(artifact) == trace["gp_artifact_sha256"]
    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    torch.testing.assert_close(saved["train_Y"][:, :2], torch.as_tensor(-F/scale))
    model = SingleTaskGP(saved["train_X"], saved["train_Y"], outcome_transform=Standardize(m=saved["train_Y"].shape[-1]))
    model.load_state_dict(saved["state_dict"]);model.eval();model.likelihood.eval()
    candidate = torch.tensor(trace["candidate_unit"], dtype=torch.double)
    posterior = model.posterior(candidate)
    np.testing.assert_allclose(posterior.mean.detach().numpy(), trace["posterior_mean"], rtol=1e-8, atol=1e-8)
    physical = prepared.lower+candidate.numpy()*(prepared.upper-prepared.lower)
    assert np.all(physical >= prepared.lower) and np.all(physical <= prepared.upper)
    actual_f, actual_g, _ = prepared.evaluate(physical)
    np.testing.assert_allclose(actual_f, trace["observed_objectives"])
    np.testing.assert_allclose(actual_g, trace["observed_constraints"])
    if name in ("MES", "JES"):
        assert trace["surrogate_function_points"] > 0 and len(trace["pareto_sets"]) == 4


@pytest.mark.parametrize("name", ["ParEGO", "NEHVI"])
def test_native_constraint_probability_changes_acquisition_preference(name):
    torch = load_tensor_runtime()
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    with tensor_random_scope(torch, 17):
        u = torch.linspace(0, 1, 9, dtype=torch.double)
        X = torch.stack((u, torch.full_like(u, 0.5)), dim=-1)
        Y = torch.stack((-(u-0.5)**2, -(u-0.5)**4, u-0.5), dim=-1)
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=3));model.eval()
        adapter = BayesianAdapter(name, None, [1, 1])
        adapter.torch, adapter.seed, adapter.n_constraints = torch, 17, 1
        bounds = torch.tensor([[0., 0.], [1., 1.]], dtype=torch.double)
        acquisition, _ = adapter._acquisition(model, X, bounds, 0)
        candidates = torch.tensor([[[0.25, 0.5]], [[0.75, 0.5]]], dtype=torch.double)
        values = acquisition(candidates)
        assert torch.isfinite(values).all() and values[0] > values[1]
