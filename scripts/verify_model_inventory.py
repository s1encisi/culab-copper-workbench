"""Verify saved inventory models, explanations and local release-source replay."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import numpy as np
import pandas as pd
from copper_mvp.access import AccessControl
from copper_mvp.common import PROJECT_ROOT, digest, file_hash, write_json
from copper_mvp.data import DataRepository
from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_evaluation import read_training
from copper_mvp.model_lifecycle import ModelLifecycle
from copper_mvp.model_training import load_registered_model
from copper_mvp.release_sources import ArtifactSources
from copper_mvp.research_store import ResearchStore
from copper_mvp.specialized_models import native_matrix
from copper_mvp.storage import RunStore


def verify_inventory(run_root, study_id, output):
    study_root = run_root / "model_inventory_studies" / study_id
    state = json.loads((study_root / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed", state["status"]
    protocol = json.loads((study_root / "protocol.json").read_text(encoding="utf-8"))
    assert file_hash(study_root / "evaluation.json") == state["evaluation_sha256"]
    evaluation = json.loads((study_root / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["protocol_sha256"] == file_hash(study_root / "protocol.json")
    assert evaluation["predictions_sha256"] == file_hash(study_root / "predictions.csv")
    for path, expected in protocol["code_hashes"].items():
        assert file_hash(study_root / "executed_source" / path) == expected
        assert file_hash(PROJECT_ROOT / path) == expected, path
    methods = [m["method_id"] for m in protocol["methods"]]
    data = DataRepository()
    comparisons = ComparisonService(run_root / "model_comparisons", data)
    report = {"status": "running", "study_id": study_id, "artifacts": [],
              "explanations": [], "release_sources": []}
    for item in state["comparisons"]:
        folder = comparisons.directory(item["comparison_id"])
        record = comparisons.get(item["comparison_id"])
        assert record["result"]["status"] == "completed"
        manifest, fit_protocol = read_training(folder)
        oof = pd.read_csv(folder / "oof_predictions.csv")
        uncertainty_path = folder / "oof_uncertainty.csv"
        uncertainty = pd.read_csv(uncertainty_path) if uncertainty_path.exists() else None
        for entry in manifest["artifacts"]:
            method, fold = entry["method_id"], entry["fold_id"]
            if method not in methods:
                continue
            model, loaded_entry = load_registered_model(folder, manifest, fit_protocol, method, fold)
            assert entry == loaded_entry and model.is_fitted
            if method == "BART":
                assert model.fit_metadata["diagnostics_passed"], (item["seed"], fold, model.fit_metadata["diagnostics"])
            evidence = {"comparison_id": item["comparison_id"], "seed": item["seed"],
                        "method": method, "fold": fold, "sha256": entry["sha256"],
                        "warnings": dict(Counter(w["category"] for w in model.fit_warnings))}
            if fold == "DEVELOPMENT":
                evidence["scope"] = "serialized_full_development_fit"
            else:
                part = oof[oof.method_id.eq(method) & oof.fold_id.eq(fold)]
                indices = np.unique(np.linspace(0, len(part)-1, min(8, len(part))).astype(int))
                sample = part.iloc[indices]
                events = sample.event_id.tolist()
                X = data.X.loc[events].to_numpy(float)
                predicted = model.predict(X)
                expected = sample[["cu", "as"]].to_numpy(float)
                np.testing.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-9)
                evidence.update(scope="held_out_oof_replay", replayed_events=len(events),
                                maximum_absolute_difference=float(np.max(np.abs(predicted-expected))))
                if method == "NGBoost":
                    distribution = model.predict_uncertainty(X)
                    np.testing.assert_allclose(distribution["mean"], expected, rtol=1e-10, atol=1e-9)
                    for column, target in enumerate(("cu", "as")):
                        expected_std = uncertainty[uncertainty.target.eq(target)].set_index("event_id").loc[events, "std"]
                        np.testing.assert_allclose(distribution["std"][:, column], expected_std, rtol=1e-10, atol=1e-9)
                if method == "BART":
                    quantiles = model.predict_uncertainty(X)["values"]
                    for column, target in enumerate(("cu", "as")):
                        expected_q = uncertainty[uncertainty.target.eq(target)].set_index("event_id").loc[events, ["q10", "q50", "q90"]].to_numpy(float)
                        np.testing.assert_allclose(quantiles[:, :, column], expected_q, rtol=1e-10, atol=1e-9)
                if method == "SymbolicRegression":
                    from copper_mvp.symbolic_expression import evaluate_expression
                    payload = model._symbolic.equations()
                    subset = X[:, payload["input_indices"]].copy()
                    imputation = np.asarray(payload["imputation_values"])
                    subset = np.where(np.isnan(subset), imputation, subset)
                    standardized = (subset-np.asarray(payload["input_mean"]))/np.asarray(payload["input_scale"])
                    standardized[:, ~np.asarray(payload["active_input_mask"], bool)] = 0.
                    delta = np.column_stack([evaluate_expression(m["ast"], standardized) for m in payload["models"]])
                    reconstructed = X[:, :2]+np.asarray(payload["target_delta_mean"])+np.asarray(payload["target_delta_scale"])*delta
                    np.testing.assert_allclose(reconstructed, predicted, rtol=1e-10, atol=1e-9)
                    path = output / "explanations" / f"Symbolic-{item['seed']}-{fold}.json"
                    write_json(path, payload)
                    report["explanations"].append({"method": method, "seed": item["seed"], "fold": fold,
                        "path": path.relative_to(output).as_posix(), "sha256": file_hash(path),
                        "affine_reconstruction": True, "expressions": [m["expression"] for m in payload["models"]],
                        "structures": [m["structure"] for m in payload["models"]],
                        "selected_features": [m["selected_features"] for m in payload["models"]]})
                if method == "EBM":
                    explained = model._specialized
                    reconstructed = np.column_stack([
                        native.eval_terms(native_matrix(X)).sum(axis=1)+native.intercept_
                        for native in model.models])
                    physical = X[:, :2]+explained.target_scaler.inverse_transform(reconstructed)
                    np.testing.assert_allclose(physical, predicted, rtol=1e-10, atol=1e-9)
                    payload = {"scope": "model_associations", "feature_names": data.feature_columns,
                               "target_delta_mean": explained.target_scaler.mean_.tolist(),
                               "target_delta_scale": explained.target_scaler.scale_.tolist(), "targets": []}
                    for target, native in zip(("cu", "as"), model.models):
                        missing_bin = [float(scores.flat[0]) for scores in native.term_scores_[:114]]
                        payload["targets"].append({"target": target, "intercept": float(native.intercept_),
                            "term_features": native.term_features_, "term_names": native.term_names_,
                            "term_importances": native.term_importances().tolist(),
                            "bins": [[v.tolist() if isinstance(v, np.ndarray) else v for v in levels] for levels in native.bins_],
                            "term_scores": [v.tolist() for v in native.term_scores_],
                            "bin_weights": [v.tolist() for v in native.bin_weights_],
                            "missing_bin_scores": missing_bin,
                            "probe_main_effects": (native.eval_terms(native_matrix(X))[:, :114]*explained.target_scaler.scale_[len(payload["targets"])]).tolist()})
                    path = output / "explanations" / f"EBM-{item['seed']}-{fold}.json"
                    write_json(path, payload)
                    report["explanations"].append({"method": method, "seed": item["seed"], "fold": fold,
                        "path": path.relative_to(output).as_posix(), "sha256": file_hash(path), "additive_reconstruction": True})
                if method == "Cubist":
                    payload = model._specialized._cubist.rules()
                    path = output / "explanations" / f"Cubist-{item['seed']}-{fold}.json"
                    write_json(path, payload)
                    # Inspect adjacent predictions across the observed feature range; do not assume global continuity.
                    probe = np.repeat(X[:1], 101, axis=0)
                    lower, upper = np.nanquantile(data.X.loc[data.training_ids(fold)].iloc[:, 2], [.05, .95])
                    probe[:, 2] = np.linspace(lower, upper, len(probe))
                    response = model.predict(probe)
                    assert np.isfinite(response).all()
                    report["explanations"].append({"method": method, "seed": item["seed"], "fold": fold,
                        "path": path.relative_to(output).as_posix(), "sha256": file_hash(path),
                        "feature_probe": data.feature_columns[2], "probe_points": len(probe),
                        "largest_adjacent_change": np.max(np.abs(np.diff(response, axis=0)), axis=0).tolist(),
                        "rule_text_lengths": [len(m) for m in payload["models"]],
                        "global_continuity_claimed": False})
            report["artifacts"].append(evidence)
        write_json(output / "verification.json", report)
        print(json.dumps({"verified_seed": item["seed"], "artifacts": len(report["artifacts"])}), flush=True)
    stability = []
    ebm_records = [r for r in report["explanations"] if r["method"] == "EBM"]
    for fold in sorted({r["fold"] for r in ebm_records}):
        payloads = [json.loads((output / r["path"]).read_text(encoding="utf-8"))
                    for r in ebm_records if r["fold"] == fold]
        for index, target in enumerate(("cu", "as")):
            terms = [p["targets"][index] for p in payloads]
            effects = np.array([t["probe_main_effects"] for t in terms])
            top = [set(np.argsort(t["term_importances"][:114])[-10:].tolist()) for t in terms]
            overlaps = [len(a & b)/len(a | b) for i, a in enumerate(top) for b in top[i+1:]]
            stability.append({"fold": fold, "target": target, "seeds": len(terms),
                "top10_main_feature_jaccard_mean": float(np.mean(overlaps)) if overlaps else None,
                "mean_probe_main_effect_seed_sd": float(np.std(effects, axis=0, ddof=1).mean()) if len(terms)>1 else None,
                "maximum_probe_main_effect_seed_sd": float(np.std(effects, axis=0, ddof=1).max()) if len(terms)>1 else None})
    report["ebm_seed_stability"] = stability
    symbolic_records = [r for r in report["explanations"] if r["method"] == "SymbolicRegression"]
    if symbolic_records:
        report["symbolic_structure_stability"] = {}
        for index, target in enumerate(("cu", "as")):
            signatures = Counter(json.dumps(r["structures"][index], sort_keys=True) for r in symbolic_records)
            features = [set(r["selected_features"][index]) for r in symbolic_records]
            overlap = [len(a & b)/len(a | b) if a | b else 1. for i,a in enumerate(features) for b in features[i+1:]]
            report["symbolic_structure_stability"][target] = {"fitted_expressions": len(features),
                "distinct_structures": len(signatures), "structure_counts": dict(signatures),
                "mean_feature_jaccard": float(np.mean(overlap)) if overlap else None}
    store = RunStore(output / "lifecycle")
    access = AccessControl(ResearchStore(store))
    actor = access.authenticate(key=access.owner_key_path.read_text().strip())
    lifecycle = ModelLifecycle(store, access, ArtifactSources(run_root, data))
    source_id = state["comparisons"][0]["comparison_id"]
    for method in methods:
        for target in ("cu", "as"):
            key = f"{method}-{target}"
            candidate = lifecycle.register(actor, {"request_key": "register-"+key,
                "source_kind": "model_comparison", "source_id": source_id,
                "method_id": method, "target": target})
            shadow = lifecycle.start_shadow(actor, candidate["id"],
                {"request_key": "shadow-"+key, "samples_per_fold": 4})
            assert shadow["status"] == "completed" and shadow["evidence"]["parity_passed"]
            report["release_sources"].append({"method": method, "target": target,
                "candidate_id": candidate["id"], "shadow_id": shadow["id"],
                "samples": shadow["evidence"]["samples"], "parity_passed": True,
                "qualification": lifecycle.qualification(actor, candidate["id"], shadow["id"])})
            write_json(output / "verification.json", report)
    pointers = lifecycle.pointers(actor)
    assert pointers["cu"]["version"] == pointers["as"]["version"] == 0
    report.update(status="PASSED", pointers=pointers, default_models_changed=False,
                  actual_replayed_events=sum(r.get("replayed_events", 0) for r in report["artifacts"]),
                  fit_artifacts=len(report["artifacts"]),
                  study_evaluation_sha256=file_hash(study_root / "evaluation.json"))
    write_json(output / "verification.json", report)
    print(json.dumps({"status": "PASSED", "fit_artifacts": report["fit_artifacts"],
                      "replayed_events": report["actual_replayed_events"],
                      "release_sources": len(report["release_sources"])}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.study_id) != 32 or any(c not in "0123456789abcdef" for c in args.study_id):
        parser.error("invalid study ID")
    for path in (args.run_root, args.output):
        if not path.resolve().is_relative_to((PROJECT_ROOT / "runs").resolve()):
            parser.error("private artifacts must remain under local runs/")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "explanations").mkdir(exist_ok=True)
    verify_inventory(args.run_root, args.study_id, args.output)
