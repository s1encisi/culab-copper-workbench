"""Independent audit of scalarization records, geometry and charged return checks."""

from __future__ import annotations

import numpy as np


def audit_scalarization(record, frame, prepared):
    detail = record["scalarization"]
    jobs = detail["jobs"]
    anchor_jobs = [job for job in jobs if job["stage"] == "anchor"]
    preference_jobs = [job for job in jobs if job["stage"] == "preference"]
    if sum(job["actual_evaluations"] for job in jobs) != record["evaluations"]["search"]:
        raise ValueError("Scalar job ledger does not cover every search evaluation")
    scale = np.asarray(record["metric"]["scale"])
    anchors = detail["anchors"]
    if anchors is not None:
        end = max(job["ledger_stop"] for job in anchor_jobs)
        prefix = frame.iloc[:end]
        prefix = prefix[prefix.phase.isin(("pilot", "initial", "search"))]
        gcols = [c for c in prefix if c.startswith("g") and c[1:].isdigit()]
        valid = prefix.loc[np.all(prefix[gcols].to_numpy() <= 1e-8, axis=1)]
        F = valid[["f0", "f1"]].to_numpy() / scale
        indices = [np.lexsort((F[:, 1 - i], F[:, i]))[0] for i in (0, 1)]
        expected = F[indices]
        ideal = np.array([expected[0, 0], expected[1, 1]])
        span = np.array([expected[1, 0] - ideal[0], expected[0, 1] - ideal[1]])
        norm = np.where(span > 1e-10, span, 1.0)
        np.testing.assert_allclose(anchors["raw_objectives"], expected * scale, atol=1e-8, rtol=1e-9)
        np.testing.assert_allclose(anchors["scaled_ideal"], ideal, atol=1e-12)
        np.testing.assert_allclose(anchors["normalization"], norm, atol=1e-12)
        if detail["status"] == "degenerate_anchor_geometry" and (
            record["optimizer_id"] not in ("NBI", "NNC") or np.all(span > 1e-10)
        ):
            raise ValueError("Invalid degeneracy classification")
    if detail["status"] == "scalarization_sweep_complete":
        if len(anchor_jobs) != 4 or len({job["parameter"] for job in preference_jobs}) < 2:
            raise ValueError("Incomplete anchor or preference sweep")
        for parameter in {job["parameter"] for job in preference_jobs}:
            if {job["restart"] for job in preference_jobs if job["parameter"] == parameter} != {0, 1}:
                raise ValueError("Missing preference restart")
    extra_evaluations = 0
    for job in jobs:
        if (
            job["ledger_stop"] - job["ledger_start"] != job["actual_evaluations"]
            or job["actual_evaluations"] > job["quota"]
        ):
            raise ValueError("Scalar job budget mismatch")
        if "returned" not in job:
            continue
        if (
            job["solver_nfev"] + job["return_verification_evaluations"] != job["actual_evaluations"]
            or job["return_verification_evaluations"] != 1
        ):
            raise ValueError("Missing or uncharged return-point check")
        returned = job["returned"]
        X = np.asarray(returned["variables"])[None, :]
        # Match the original single-vector return check, including its numerical path.
        F, G, _ = prepared.evaluate(X)
        extra_evaluations += 1
        np.testing.assert_allclose(F[0], returned["objectives"], rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(G[0], returned["constraints"], rtol=1e-9, atol=1e-8)
        row = frame.iloc[job["ledger_stop"] - 1]
        xcols = [f"x{i}" for i in range(X.shape[1])]
        np.testing.assert_allclose(row[xcols].to_numpy(float), X[0], rtol=1e-12, atol=1e-10)
        coordinates = np.asarray(returned["solver_coordinates"])
        if job["stage"] == "anchor":
            value, additional = F[0, int(job["parameter"])] / scale[int(job["parameter"])], np.empty(0)
        else:
            y = (F[0] / scale - np.asarray(anchors["scaled_ideal"])) / np.asarray(anchors["normalization"])
            p = job["parameter"]
            name = record["optimizer_id"]
            if name == "Weighted-Sum":
                value, additional = (1 - p) * y[0] + p * y[1], np.empty(0)
            elif name == "Augmented-Chebyshev":
                terms = np.array([(1 - p) * y[0], p * y[1]])
                value, additional = terms.max() + 0.001 * terms.sum(), np.empty(0)
            elif name == "Epsilon-Constraint":
                value, additional = y[1], np.array([y[0] - p])
            elif name == "NNC":
                value, additional = y[1], np.array([y[0] - y[1] - (2 * p - 1)])
            elif name == "NBI":
                residual = y - np.array([p, 1 - p]) + coordinates[-1] / np.sqrt(2)
                value, additional = -coordinates[-1], np.r_[residual - 1e-6, -residual - 1e-6]
            else:
                raise ValueError("Unsupported scalar audit method")
        np.testing.assert_allclose(additional, returned["preference_constraints"], atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(value, returned["scalar_objective"], atol=1e-10, rtol=1e-9)
        unit = coordinates[: X.shape[1]]
        maxcv = float(np.r_[0.0, G[0], additional, -unit, unit - 1].max())
        np.testing.assert_allclose(maxcv, job["maxcv"], atol=1e-10, rtol=1e-9)
        if job["subproblem_feasible"] and maxcv > 1e-8 + 1e-10:
            raise ValueError("Reported feasible scalar return violates its constraints")
        if job["solver_converged"] and (
            job["solver_info"] not in (0, 1) or not job["subproblem_feasible"] or not job["solver_return_consistent"]
        ):
            raise ValueError("Invalid scalar convergence claim")
    return {
        "jobs": len(jobs),
        "converged": sum(job["solver_converged"] for job in jobs),
        "extra_return_evaluations": extra_evaluations,
        "status": detail["status"],
    }
