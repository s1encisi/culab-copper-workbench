"""Independent reconstruction of saved Bayesian optimizer decisions."""

from pathlib import Path

import numpy as np

from copper_mvp.common import file_hash
from copper_mvp.neural_runtime import load_tensor_runtime, tensor_random_scope


def audit_bayesian(record, frame, prepared, folder):
    torch = load_tensor_runtime()
    from botorch.acquisition.multi_objective.joint_entropy_search import qLowerBoundMultiObjectiveJointEntropySearch
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.max_value_entropy_search import (
        qLowerBoundMultiObjectiveMaxValueEntropySearch,
    )
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.acquisition.multi_objective.parego import qLogNParEGO
    from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.sampling.normal import SobolQMCNormalSampler

    iterations = record["bayesian"]["iterations"]
    if sum(row.get("evaluated_candidates", 0) for row in iterations) != record["evaluations"]["search"]:
        raise ValueError("BO iteration costs do not match the physical ledger")
    extra, checked, artifacts = 0, 0, {}
    scale = np.array(record["metric"]["scale"])
    xcols = [f"x{i}" for i in range(len(prepared.lower))]
    gcols = [f"g{i}" for i in range(prepared.constraints)]
    for row in iterations:
        before, after = row["physical_evaluations_before"], row["physical_evaluations_after"]
        if after - before != row.get("evaluated_candidates", 0):
            raise ValueError("BO iteration ledger boundary mismatch")
        if row["status"] != "evaluated":
            continue
        artifact = (Path(folder) / row["gp_artifact"]).resolve()
        if not artifact.is_relative_to(Path(folder).resolve()) or file_hash(artifact) != row["gp_artifact_sha256"]:
            raise ValueError("GP artifact path or hash mismatch")
        saved = torch.load(artifact, map_location="cpu", weights_only=True)
        past = frame.iloc[:before]
        past = past[past.phase.isin(("pilot", "initial", "search"))]
        indices = np.asarray(row["training_ledger_indices"], dtype=int)
        expected_indices = (
            np.arange(len(past)) if len(past) <= 256 else np.r_[np.arange(64), np.arange(len(past) - 192, len(past))]
        )
        np.testing.assert_array_equal(indices, expected_indices)
        selected = past.iloc[indices]
        X = (selected[xcols].to_numpy() - prepared.lower) / (prepared.upper - prepared.lower)
        Y = np.column_stack((-selected[["f0", "f1"]].to_numpy() / scale, selected[gcols].to_numpy() - 1e-8))
        np.testing.assert_allclose(saved["train_X"].numpy(), X, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(saved["train_Y"].numpy(), Y, rtol=1e-10, atol=1e-10)
        model = SingleTaskGP(saved["train_X"], saved["train_Y"], outcome_transform=Standardize(m=Y.shape[1]))
        model.load_state_dict(saved["state_dict"])
        model.eval()
        model.likelihood.eval()
        candidate = torch.tensor(row["candidate_unit"], dtype=torch.double)
        if not torch.all((candidate >= 0) & (candidate <= 1)):
            raise ValueError("BO candidate outside its unit box")
        with torch.no_grad():
            posterior = model.posterior(candidate)
            np.testing.assert_allclose(posterior.mean.numpy(), row["posterior_mean"], rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(posterior.variance.numpy(), row["posterior_variance"], rtol=1e-6, atol=1e-9)
        physical = prepared.lower + candidate.numpy() * (prepared.upper - prepared.lower)
        np.testing.assert_allclose(frame.iloc[before:after][xcols].to_numpy(), physical, rtol=1e-12, atol=1e-10)
        F, G, _ = prepared.evaluate(physical)
        extra += len(physical)
        np.testing.assert_allclose(F, row["observed_objectives"], rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(G, row["observed_constraints"], rtol=1e-9, atol=1e-8)
        if "acquisition_rng_state" in saved:
            with tensor_random_scope(torch, saved["iteration_seed"]), torch.no_grad():
                torch.random.set_rng_state(saved["acquisition_rng_state"])
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=record["seed"] + row["iteration"])
                objective = IdentityMCMultiOutputObjective(outcomes=[0, 1])
                constraints = [lambda samples, i=i: samples[..., i] for i in range(2, Y.shape[1])]
                name = record["optimizer_id"]
                if name == "ParEGO":
                    weights = torch.distributions.Dirichlet(torch.ones(2, dtype=torch.double)).sample()
                    np.testing.assert_allclose(weights.numpy(), row["preference_weights"], rtol=1e-12, atol=1e-12)
                    acq = qLogNParEGO(
                        model,
                        saved["train_X"],
                        scalarization_weights=weights,
                        sampler=sampler,
                        objective=objective,
                        constraints=constraints or None,
                        prune_baseline=False,
                        cache_root=True,
                    )
                elif name == "NEHVI":
                    acq = qLogNoisyExpectedHypervolumeImprovement(
                        model,
                        ref_point=[-1.1, -1.1],
                        X_baseline=saved["train_X"],
                        sampler=sampler,
                        objective=objective,
                        constraints=constraints or None,
                        prune_baseline=False,
                        cache_root=True,
                    )
                else:
                    if prepared.constraints:
                        raise ValueError("Entropy search was used on a constrained problem")
                    pareto_x = torch.tensor(row["pareto_sets"], dtype=torch.double)
                    pareto_y = torch.tensor(row["pareto_fronts"], dtype=torch.double)
                    if not torch.all((pareto_x >= 0) & (pareto_x <= 1)):
                        raise ValueError("Posterior Pareto set outside bounds")
                    boxes = compute_sample_box_decomposition(pareto_y, maximize=True)
                    if name == "MES":
                        acq = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                            model, hypercell_bounds=boxes, estimation_type="LB", num_samples=32
                        )
                    else:
                        acq = qLowerBoundMultiObjectiveJointEntropySearch(
                            model,
                            pareto_sets=pareto_x,
                            pareto_fronts=pareto_y,
                            hypercell_bounds=boxes,
                            estimation_type="LB",
                            num_samples=32,
                        )
                if type(acq).__name__ != row["acquisition_class"]:
                    raise ValueError("Saved acquisition mechanism mismatch")
                value = float(acq(candidate))
                np.testing.assert_allclose(value, row["acquisition_value"], rtol=1e-5, atol=1e-6)
        artifacts[row["gp_artifact"]] = row["gp_artifact_sha256"]
        checked += 1
    return {
        "checked_iterations": checked,
        "extra_candidate_evaluations": extra,
        "gp_artifacts": artifacts,
        "surrogate_function_points": record["bayesian"]["surrogate_function_points"],
    }
