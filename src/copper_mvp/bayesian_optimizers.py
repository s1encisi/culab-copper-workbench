"""Four fixed Bayesian optimizer families backed by fitted BoTorch GPs."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import warnings
import numpy as np

from copper_mvp.common import WorkbenchError, file_hash, safe, write_json
from copper_mvp.neural_runtime import load_tensor_runtime, tensor_random_scope

BAYESIAN_OPTIMIZERS = ("ParEGO", "NEHVI", "MES", "JES")
ENTROPY_OPTIMIZERS = ("MES", "JES")


def bayesian_spec(name):
    mechanisms = {
        "ParEGO": ("O22", "qLogNParEGO with sampled Chebyshev preferences and GP uncertainty"),
        "NEHVI": ("O23", "qLogNoisyExpectedHypervolumeImprovement"),
        "MES": ("O24", "lower-bound multi-objective max-value entropy search"),
        "JES": ("O25", "lower-bound multi-objective joint set/front entropy search"),
    }
    design_id, mechanism = mechanisms[name]
    return {"optimizer_id": name, "design_id": design_id, "method_version": "g6f.fixed.v3", "mechanism": mechanism,
        "package": "botorch", "package_version": "0.17.2", "variables": "continuous", "objectives": 2,
        "inequality_constraints": name not in ENTROPY_OPTIMIZERS,
        "constraint_handling": "native_GP_constraint_probability" if name not in ENTROPY_OPTIMIZERS else "unconstrained_mathematical_only",
        "parameters": {"device": "cpu", "dtype": "float64", "batch_size": 4, "initial_samples": 64,
            "gp": "independent-output SingleTaskGP with native RBF priors and Standardize", "gp_max_iterations": 25,
            "gp_fit_timeout_seconds": 15, "gp_training_limit": 256, "training_window": "first_64_plus_latest_192",
            "likelihood_noise": "learned_internal_surrogate_noise", "objective_sign": "negated_shared_scaled_minimization_objectives",
            "mc_samples": 32, "acquisition_restarts": 4, "acquisition_raw_samples": 64, "acquisition_max_iterations": 20,
            "pareto_function_samples": 4, "pareto_candidate_pool": 256, "pareto_points_max": 8,
            "pareto_points_policy": "minimum_available_count_across_function_draws_without_duplicate_padding",
            "entropy_estimator": "LB", "fixed_maximization_reference": [-1.1, -1.1]},
        "status": "registered", "execution_authorized": False}


class _BayesianTimeStop(Exception):
    pass


class BayesianAdapter:
    def __init__(self, name, population, scale):
        self.name, self.initial, self.scale = name, population, np.asarray(scale)
        self.n_offsprings = 4
        self.initialized = False
        self.stop_reason = None
        self.deadline = float("inf")
        self.artifact_dir = None
        self.trace = []

    def setup(self, problem, termination=None, seed=None, verbose=False):
        self.torch = load_tensor_runtime()
        self.seed = int(seed)
        self.evaluator = problem.evaluator
        self.lower, self.span = problem.xl.copy(), problem.xu-problem.xl
        self.n_var, self.n_constraints = problem.n_var, problem.n_ieq_constr
        if self.name in ENTROPY_OPTIMIZERS and self.n_constraints:
            raise WorkbenchError("MES/JES 首版仅支持明确的无约束数学问题", "OPTIMIZER_CAPABILITY")
        self.limit = self.evaluator.budget-min(128, max(16, self.evaluator.budget//8))

    def has_next(self):
        return self.stop_reason is None

    def _remaining_time(self):
        remaining = self.deadline-perf_counter()
        if remaining <= 0:
            raise _BayesianTimeStop()
        return remaining

    def _training_data(self):
        torch = self.torch
        X, F, G = self.evaluator.search_points()
        indices = np.arange(len(X)) if len(X) <= 256 else np.r_[np.arange(64), np.arange(len(X)-192, len(X))]
        target = np.column_stack((-F/self.scale, G-1e-8))
        return (torch.as_tensor((X[indices]-self.lower)/self.span, dtype=torch.double),
                torch.as_tensor(target[indices], dtype=torch.double), indices)

    def _sample_fronts(self, model, bounds, train_x):
        torch = self.torch
        from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.sampling import draw_sobol_samples
        points, fronts = [], []
        with torch.no_grad():
            for _ in range(4):
                self._remaining_time()
                sample = get_matheron_path_model(model)
                grid = torch.cat([draw_sobol_samples(bounds, n=256, q=1).squeeze(-2), train_x])
                values = sample.posterior(grid).mean
                keep = is_non_dominated(values)
                points.append(grid[keep]); fronts.append(values[keep])
            count = min(8, min(len(front) for front in fronts))
            if count < 1:
                raise WorkbenchError("未获得有效的后验 Pareto 样本", "BAYESIAN_PARETO_SAMPLE")
            chosen_x, chosen_y = [], []
            for x, y in zip(points, fronts):
                order = torch.argsort(y[:, 0])
                selected = torch.linspace(0, len(order)-1, count).long()
                chosen_x.append(x[order[selected]]); chosen_y.append(y[order[selected]])
        return torch.stack(chosen_x), torch.stack(chosen_y), 4*(256+len(train_x))

    def _acquisition(self, model, train_x, bounds, iteration):
        torch = self.torch
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=self.seed+iteration)
        constraints = [lambda samples, index=index: samples[..., index] for index in range(2, 2+self.n_constraints)]
        if self.name == "ParEGO":
            from botorch.acquisition.multi_objective.parego import qLogNParEGO
            weights = torch.distributions.Dirichlet(torch.ones(2, dtype=torch.double)).sample()
            acq = qLogNParEGO(model, train_x, scalarization_weights=weights, sampler=sampler,
                objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]), constraints=constraints or None,
                prune_baseline=False, cache_root=True)
            return acq, {"preference_weights": weights.tolist(), "surrogate_function_points": 0}
        if self.name == "NEHVI":
            from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
            acq = qLogNoisyExpectedHypervolumeImprovement(model, ref_point=[-1.1, -1.1], X_baseline=train_x,
                sampler=sampler, objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
                constraints=constraints or None, prune_baseline=False, cache_root=True)
            return acq, {"surrogate_function_points": 0}
        from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
        pareto_x, pareto_y, evaluations = self._sample_fronts(model, bounds, train_x)
        boxes = compute_sample_box_decomposition(pareto_y, maximize=True)
        if self.name == "MES":
            from botorch.acquisition.multi_objective.max_value_entropy_search import qLowerBoundMultiObjectiveMaxValueEntropySearch
            acq = qLowerBoundMultiObjectiveMaxValueEntropySearch(model, hypercell_bounds=boxes, estimation_type="LB", num_samples=32)
        else:
            from botorch.acquisition.multi_objective.joint_entropy_search import qLowerBoundMultiObjectiveJointEntropySearch
            acq = qLowerBoundMultiObjectiveJointEntropySearch(model, pareto_sets=pareto_x, pareto_fronts=pareto_y,
                                                            hypercell_bounds=boxes, estimation_type="LB", num_samples=32)
        return acq, {"surrogate_function_points": evaluations, "pareto_sets": pareto_x.tolist(),
                     "pareto_fronts": pareto_y.tolist(), "hypercell_count": boxes.shape[-2]}

    def next(self):
        if not self.initialized:
            self.initialized = True
            return
        torch = self.torch
        iteration = len(self.trace)
        detail = {"iteration": iteration, "physical_evaluations_before": self.evaluator.used,
                  "acquisition_calls": 0, "acquisition_candidate_points": 0}
        started = perf_counter()
        notices = []
        try:
            with tensor_random_scope(torch, self.seed+104729*iteration), warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                from botorch.models import SingleTaskGP
                from botorch.models.transforms.outcome import Standardize
                from botorch.optim.fit import fit_gpytorch_mll_scipy
                from botorch.optim import optimize_acqf
                from gpytorch.mlls import ExactMarginalLogLikelihood
                train_x, train_y, indices = self._training_data()
                detail.update(training_rows=len(train_x), training_ledger_indices=indices.tolist())
                model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
                detail.update(gp_class=type(model).__name__, gp_kernel=type(model.covar_module).__name__,
                              modeled_constraint_outputs=self.n_constraints)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_start = perf_counter()
                fit = fit_gpytorch_mll_scipy(mll, options={"maxiter": 25}, timeout_sec=min(15.0, self._remaining_time()))
                detail.update(gp_fit_ms=(perf_counter()-fit_start)*1000, gp_fit_status=str(fit.status),
                              gp_fit_steps=int(fit.step), gp_fit_value=float(fit.fval))
                if not np.isfinite(fit.fval) or "FAILURE" in str(fit.status):
                    raise WorkbenchError("贝叶斯代理拟合失败", "BAYESIAN_GP_FIT")
                model.eval(); model.likelihood.eval()
                bounds = torch.stack([torch.zeros(self.n_var, dtype=torch.double), torch.ones(self.n_var, dtype=torch.double)])
                acquisition_rng_state = torch.random.get_rng_state().clone()
                acq, acquisition_detail = self._acquisition(model, train_x, bounds, iteration)
                detail.update(acquisition_detail)
                detail["acquisition_class"] = type(acq).__name__
                def before_forward(module, inputs):
                    self._remaining_time()
                    X = inputs[0]
                    detail["acquisition_calls"] += 1
                    detail["acquisition_candidate_points"] += int(X.numel()//self.n_var)
                hook = acq.register_forward_pre_hook(before_forward)
                q = min(4, self.n_offsprings, self.limit-self.evaluator.used)
                acq_start = perf_counter()
                try:
                    candidate, value = optimize_acqf(acq, bounds, q=q, num_restarts=4, raw_samples=64,
                        options={"batch_limit": 1, "maxiter": 20, "seed": self.seed+iteration},
                        timeout_sec=self._remaining_time(), retry_on_optimization_warning=False)
                finally:
                    hook.remove()
                detail["acquisition_ms"] = (perf_counter()-acq_start)*1000
                self._remaining_time()
                candidate = candidate.detach().clamp(0, 1)
                if not torch.isfinite(candidate).all() or not torch.isfinite(value).all():
                    raise WorkbenchError("采集函数返回非有限候选", "BAYESIAN_ACQUISITION")
                with torch.no_grad():
                    checked_value = acq(candidate)
                    if not torch.isfinite(checked_value).all():
                        raise WorkbenchError("候选处采集函数值非有限", "BAYESIAN_ACQUISITION")
                    detail["acquisition_calls"] += 1
                    detail["acquisition_candidate_points"] += q
                    prediction = model.posterior(candidate)
                    detail.update(candidate_unit=candidate.tolist(), acquisition_value=float(checked_value),
                        optimizer_reported_acquisition_value=float(value),
                        posterior_mean=prediction.mean.tolist(), posterior_variance=prediction.variance.tolist())
                if self.artifact_dir is not None:
                    target = Path(self.artifact_dir)/"gp_models"/f"iteration_{iteration:03d}.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"train_X": train_x, "train_Y": train_y, "state_dict": model.state_dict(),
                                "iteration_seed": self.seed+104729*iteration, "acquisition_rng_state": acquisition_rng_state}, target)
                    detail.update(gp_artifact=str(target.relative_to(self.artifact_dir)).replace("\\", "/"), gp_artifact_sha256=file_hash(target))
                physical = self.lower+candidate.cpu().numpy()*self.span
                F, G, _ = self.evaluator.evaluate(physical, "search")
                detail.update(observed_objectives=F.tolist(), observed_constraints=G.tolist(), evaluated_candidates=len(physical), status="evaluated")
                detail["warnings"] = [{"category": type(w.message).__name__, "message": str(w.message)} for w in notices]
        except _BayesianTimeStop:
            self.stop_reason = "time_budget"
            detail["status"] = "time_budget_before_physical_evaluation"
        except Exception as exc:
            detail.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__))
            raise
        finally:
            detail["warnings"] = [{"category": type(w.message).__name__, "message": str(w.message)} for w in notices]
            detail.update(physical_evaluations_after=self.evaluator.used, elapsed_ms=(perf_counter()-started)*1000)
            self.trace.append(safe(detail))
            if self.artifact_dir is not None:
                write_json(Path(self.artifact_dir)/"bo_iterations"/f"iteration_{iteration:03d}.json", detail)

    def details(self):
        return safe({"status": self.stop_reason or "total_budget", "iterations": self.trace,
            "actual_search_evaluations": sum(r.get("evaluated_candidates", 0) for r in self.trace),
            "surrogate_function_points": sum(r.get("surrogate_function_points", 0) for r in self.trace),
            "scope": "optimizer_surrogate_uncertainty; not laboratory uncertainty or a causal process model"})
