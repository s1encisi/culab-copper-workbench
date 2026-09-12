"""Sample independent BART chains and export trees paired with posterior noise."""
from collections import Counter
from pathlib import Path
import time
import warnings
import numpy as np
from copper_mvp.bart_trees import export_posterior, draw_predictions
from copper_mvp.common import write_json


def _seed_compiled_random(seed):
    np.random.seed(seed)


def fit_bart_posterior(X, y, settings, seed, output, runtime):
    import arviz as az
    import pymc as pm
    import pymc_bart as pmb
    import pytensor
    assert str(pytensor.config.mode) == settings["pytensor_mode"]
    runtime = {**runtime, "pytensor_mode": str(pytensor.config.mode)}
    from numba import njit
    seed_compiled = njit(cache=True)(_seed_compiled_random)
    output = Path(output)
    probe = np.unique(np.linspace(0, len(X)-1, min(8, len(X))).astype(int))
    targets, all_notices = [], []
    start = time.perf_counter()
    for target, name in enumerate(("cu", "as")):
        chains, traces, parity = [], [], []
        for chain in range(settings["chains"]):
            chain_seed = (seed+10000*target+chain) % (2**31)
            # PGBART initializes NumPy random buffers before pm.sample seeds its loop.
            np.random.seed(chain_seed)
            seed_compiled(chain_seed)
            write_json(output/"state.json", {"status": "sampling", "target": name,
                       "chain": chain, "seed": chain_seed, "elapsed_seconds": time.perf_counter()-start})
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                with pm.Model() as model:
                    mu = pmb.BART("mu", X, y[:, target], m=settings["trees"],
                                  alpha=settings["alpha"], beta=settings["beta"])
                    sigma = pm.HalfNormal("sigma", settings["sigma_prior_scale"])
                    pm.Normal("observed", mu=mu, sigma=sigma, observed=y[:, target])
                    tree_step = pmb.PGBART(vars=[mu], num_particles=settings["particles"],
                        batch=(settings["batch_fraction"], settings["batch_fraction"]))
                    noise_step = pm.NUTS(vars=[sigma], target_accept=settings["target_accept"])
                    manager = mu.owner.op.all_trees._manager
                    try:
                        def update_sampling(trace, draw):
                            index = draw.draw_idx+1
                            if index % 100 == 0 or index == settings["tune"]+settings["draws"]:
                                write_json(output/"state.json", {"status": "sampling", "target": name,
                                    "chain": chain, "seed": chain_seed, "iterations": index,
                                    "total_iterations": settings["tune"]+settings["draws"],
                                    "tuning": bool(draw.tuning), "elapsed_seconds": time.perf_counter()-start})
                        trace = pm.sample(draws=settings["draws"], tune=settings["tune"],
                            chains=1, cores=1, random_seed=chain_seed,
                            step=[tree_step, noise_step], progressbar=False,
                            compute_convergence_checks=False, callback=update_sampling)
                        collections = list(mu.owner.op.all_trees)
                        if len(collections) != settings["draws"]:
                            raise ValueError("Retained BART trees do not match posterior draws")
                        posterior = export_posterior(collections)
                        reconstructed = draw_predictions(posterior, X[probe])
                        expected = trace.posterior["mu"].values[0][:, probe]
                        np.testing.assert_allclose(reconstructed, expected, rtol=1e-10, atol=1e-9)
                        posterior["sigma"] = trace.posterior["sigma"].values[0].copy()
                        posterior["seed"] = chain_seed
                        posterior["probe_mu"] = expected.copy()
                        chains.append(posterior)
                        traces.append(trace)
                        parity.append(float(np.max(np.abs(reconstructed-expected))))
                    finally:
                        manager.shutdown()
            all_notices.extend({"category": w.category.__name__, "message": str(w.message)[:400]} for w in notices)
        combined = az.concat(traces, dim="chain")
        diagnostic_data = az.from_dict(posterior={
            "sigma": np.stack([c["sigma"] for c in chains]),
            "probe_mu": np.stack([c["probe_mu"] for c in chains])})
        # Compute unrounded diagnostics rather than using presentation-rounded az.summary.
        rhat = az.rhat(diagnostic_data)
        bulk = az.ess(diagnostic_data, method="bulk")
        tail = az.ess(diagnostic_data, method="tail")
        diagnostic_rows = {}
        for variable in ("sigma", "probe_mu"):
            diagnostic_rows[variable] = {
                "rhat": np.asarray(rhat[variable]).tolist(),
                "ess_bulk": np.asarray(bulk[variable]).tolist(),
                "ess_tail": np.asarray(tail[variable]).tolist()}
        maximum_rhat = max(float(np.nanmax(np.asarray(rhat[v]))) for v in ("sigma", "probe_mu"))
        minimum_bulk = min(float(np.nanmin(np.asarray(bulk[v]))) for v in ("sigma", "probe_mu"))
        minimum_tail = min(float(np.nanmin(np.asarray(tail[v]))) for v in ("sigma", "probe_mu"))
        divergences = int(combined.sample_stats["diverging"].sum())
        passed = (np.isfinite(maximum_rhat) and maximum_rhat <= settings["maximum_rhat"]
                  and minimum_bulk >= settings["minimum_ess"] and minimum_tail >= settings["minimum_ess"]
                  and divergences == 0)
        diagnostics = {"passed": bool(passed), "maximum_rhat": maximum_rhat,
            "minimum_bulk_ess": minimum_bulk, "minimum_tail_ess": minimum_tail,
            "divergences": divergences, "variables": diagnostic_rows,
            "training_probe_indices": probe.tolist(), "maximum_tree_trace_error": max(parity)}
        targets.append({"target": name, "chains": chains, "diagnostics": diagnostics})
        write_json(output/(name+"_diagnostics.json"), diagnostics)
    counts = Counter((n["category"], n["message"]) for n in all_notices)
    notices = [{"category": k[0], "message": k[1], "count": count} for k,count in counts.items()]
    if not all(t["diagnostics"]["passed"] for t in targets):
        notices.append({"category": "BARTSamplingDiagnostics",
                        "message": "Sampling diagnostics do not meet the declared research acceptance thresholds.",
                        "count": 1})
    return {"targets": targets, "settings": settings, "seed": seed, "runtime": runtime,
            "fit_warnings": notices, "elapsed_seconds": time.perf_counter()-start,
            "diagnostics_passed": all(t["diagnostics"]["passed"] for t in targets)}
