"""Independently audit saved optimizer comparisons and pair results within cases."""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from copper_mvp.common import DEFAULT_RUNS_DIR, digest, file_hash, write_json
from copper_mvp.data import DataRepository
from copper_mvp.modeling import ModelManager
from copper_mvp.bayesian_problems import build_comparison_problem
from copper_mvp.optimizer_registry import OptimizerComparisonRequest
from copper_mvp.scalarization_audit import audit_scalarization
from copper_mvp.bayesian_audit import audit_bayesian


def hv2d(front, reference):
    previous, value = reference[1], 0.0
    for x, y in front[np.argsort(front[:, 0])]:
        if x <= reference[0] and y < previous:
            value += (reference[0] - x) * (previous - y)
            previous = y
    return float(value)


def audit_comparison(run_id):
    started = perf_counter()
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Expected a comparison ID")
    root = DEFAULT_RUNS_DIR / "optimizer_comparisons" / run_id
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if state["status"] != "completed" or comparison["status"] != "completed":
        raise ValueError("Comparison is not completed")
    request = OptimizerComparisonRequest.model_validate(protocol["request"])
    data = DataRepository() if request.mode == "plant" else None
    models = ModelManager(DEFAULT_RUNS_DIR, data) if data else None
    rows, extra_evaluations, hashes = [], 0, {}
    scalar_audits, scalar_evaluations = [], 0
    bayesian_audits, bayesian_evaluations = [], 0
    prepared_cases = {}
    for record in comparison["results"]:
        case, seed, name = record["case"], record["seed"], record["optimizer_id"]
        folder = root / f"case_{case}" / str(seed) / name
        saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if saved != {k: v for k, v in record.items() if k not in ("case", "event_id")}:
            raise ValueError("Summary differs from saved run")
        problem = json.loads((folder / "problem.json").read_text(encoding="utf-8"))
        if problem["signature"] != digest({k: v for k, v in problem.items() if k != "signature"}):
            raise ValueError("Problem signature mismatch")
        frame = pd.read_csv(folder / "evaluations.csv", float_precision="round_trip")
        if len(frame) != saved["total_evaluations"] or len(frame) > request.total_budget:
            raise ValueError("Evaluation count differs from budget ledger")
        for phase, count in saved["evaluations"].items():
            if int(frame.phase.eq(phase).sum()) != count:
                raise ValueError("Phase count differs from ledger")
        if case not in prepared_cases:
            prepared_cases[case] = build_comparison_problem(data, models, request, record["event_id"], seed)
        prepared = prepared_cases[case]
        for key, value in prepared.specification().items():
            if problem.get(key) != value:
                raise ValueError("Current physical problem differs from saved problem: " + key)
        if "scalarization" in saved and saved["algorithm_executed"]:
            scalar_audit = audit_scalarization(saved, frame, prepared)
            scalar_evaluations += scalar_audit["extra_return_evaluations"]
            scalar_audits.append({"case": case, "seed": seed, "optimizer": name, **scalar_audit})
        if "bayesian" in saved and saved["algorithm_executed"]:
            with warnings.catch_warnings(record=True) as audit_notices:
                warnings.simplefilter("always")
                checked_bo = audit_bayesian(saved, frame, prepared, folder)
            checked_bo["warning_counts"] = dict(Counter(type(w.message).__name__ for w in audit_notices))
            checked_bo["warning_messages"] = sorted({str(w.message) for w in audit_notices})
            bayesian_evaluations += checked_bo["extra_candidate_evaluations"]
            bayesian_audits.append({"case": case, "seed": seed, "optimizer": name, **checked_bo})
        candidates = saved["candidates"]
        X = np.array([[c["variables"][v["name"]] for v in problem["variables"]] for c in candidates])
        F = np.array([[c["f1"], c["f2"]] for c in candidates]).reshape(-1, 2)
        if candidates:
            actual_f, actual_g, _ = prepared.evaluate(X)
            extra_evaluations += len(X)
            np.testing.assert_allclose(actual_f, F, rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(actual_g, np.array([c["constraints"] for c in candidates]), rtol=1e-9, atol=1e-8)
            if np.any(actual_g > 1e-8) or np.any(X < prepared.lower - 1e-8) or np.any(X > prepared.upper + 1e-8):
                raise ValueError("Independent front feasibility check failed")
        normal = F / np.array(saved["metric"]["scale"])
        if not np.isclose(hv2d(normal, np.array([1.1, 1.1])), saved["hv"], rtol=1e-10, atol=1e-12):
            raise ValueError("Independent hypervolume check failed")
        gcols = [c for c in frame if c.startswith("g") and c[1:].isdigit()]
        violation = np.maximum(frame[gcols].to_numpy() - 1e-8, 0).sum(axis=1)
        rows.append({"case": case, "seed": seed, "optimizer": name, "hv": saved["hv"], "elapsed_ms": saved["elapsed_ms"],
                     "initial_population_hash": saved["initial_population_hash"], "problem_signature": saved["problem_signature"],
                     "total_evaluations": saved["total_evaluations"], "feasible_rate": saved["feasible_rate"],
                     "decision_front_points": saved["decision_front_points"], "objective_front_points": saved["objective_front_points"],
                     "verified_front_points": saved["verified_front_points"], "max_normalized_constraint_violation": float(violation.max()),
                     "objective_minima": F.min(axis=0).tolist() if candidates else None,
                     "stop_reason": saved["stop_reason"], "algorithm_executed": saved["algorithm_executed"]})
        for filename in ("problem.json", "result.json", "evaluations.csv"):
            hashes[(folder / filename).relative_to(root).as_posix()] = file_hash(folder / filename)
    bases = {(r["case"], r["seed"]): r for r in rows if r["optimizer"] == "NSGA-II"}
    paired = {}
    for name in request.optimizers:
        group = [r for r in rows if r["optimizer"] == name]
        if len(group) != len(bases):
            raise ValueError("Missing paired run")
        effects, ratios, wins, ties, losses = [], [], 0, 0, 0
        for row in group:
            base = bases[(row["case"], row["seed"])]
            if row["problem_signature"] != base["problem_signature"] or row["initial_population_hash"] != base["initial_population_hash"]:
                raise ValueError("Paired problem or initial population mismatch")
            delta = row["hv"] - base["hv"]
            tolerance = max(1e-10, base["hv"] * 1e-8)
            wins += int(delta > tolerance); losses += int(delta < -tolerance); ties += int(abs(delta) <= tolerance)
            if base["hv"] > 0:
                effects.append((row["case"], delta / base["hv"]))
            ratios.append(row["elapsed_ms"] / base["elapsed_ms"])
        cases = sorted({case for case, _ in effects})
        blocks = [[v for case, v in effects if case == selected] for selected in cases]
        ci = None
        if len(cases) > 1:
            rng = np.random.default_rng(20260912)
            bootstrap = [np.median([v for i in rng.integers(0, len(blocks), len(blocks)) for v in blocks[i]]) for _ in range(2000)]
            ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        paired[name] = {"pairs": len(group), "wins": wins, "ties": ties, "losses": losses,
                        "relative_hv_median": float(np.median([v for _, v in effects])) if effects else None,
                        "relative_hv_case_block_95ci": ci, "relative_hv_pairs_with_positive_baseline": len(effects),
                        "elapsed_ratio_median": float(np.median(ratios)),
                        "elapsed_ms_p50": float(np.quantile([r["elapsed_ms"] for r in group], 0.5)),
                        "elapsed_ms_p95": float(np.quantile([r["elapsed_ms"] for r in group], 0.95))}
    result = {"schema_version": "optimizer-independent-audit.g6f.v1", "run_id": run_id, "status": "passed", "runs": len(rows),
              "request": request.model_dump(mode="json"), "paired_baseline": "NSGA-II", "paired": paired, "rows": rows,
              "bootstrap": {"unit": "case_block_all_seeds_retained", "resamples": 2000, "seed": 20260912},
              "total_solver_evaluations": sum(r["total_evaluations"] for r in rows),
              "extra_independent_front_evaluations": extra_evaluations, "extra_problem_reference_evaluations": len(prepared_cases),
              "extra_scalar_return_evaluations": scalar_evaluations, "scalarization_audits": scalar_audits,
              "extra_bayesian_candidate_evaluations": bayesian_evaluations, "bayesian_audits": bayesian_audits,
              "audit_elapsed_ms": (perf_counter() - started) * 1000, "source_hashes": hashes,
              "auditor_sha256": file_hash(Path(__file__)),
              "auditor_dependencies": {name: file_hash(Path(__file__).resolve().parents[1]/"src/copper_mvp"/name) for name in ("scalarization_audit.py", "bayesian_audit.py")}, "automatic_promotion": False}
    write_json(root / "independent_audit.json", result)
    return {k: v for k, v in result.items() if k not in ("rows", "source_hashes", "scalarization_audits", "bayesian_audits")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    arguments = parser.parse_args()
    print(json.dumps(audit_comparison(arguments.run_id), ensure_ascii=False, indent=2))
