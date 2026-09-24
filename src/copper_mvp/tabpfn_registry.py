"""Fixed local TabPFN-v2 context model with isolated runtime requirements."""

from copper_mvp.common import PROJECT_ROOT, digest, file_hash

TABPFN_METHODS = ("TabPFN",)
TABPFN_VERSION = "g6l.tabpfn.v2"
TABPFN_PARAMETERS = {
    "components": 8,
    "n_estimators": 2,
    "fit_mode": "fit_with_cache",
    "memory_saving_mode": True,
    "maximum_context_rows": 10000,
    "seed_policy": "fixed",
}
WEIGHT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
TABPFN_FILES = tuple(
    "src/copper_mvp/" + n
    for n in ("tabpfn_registry.py", "tabpfn_runtime.py", "tabpfn_models.py", "tabpfn_client.py", "tabpfn_attention.py")
) + ("scripts/run_tabpfn_worker.py", "requirements-tabpfn-isolated.txt", "configs/runtime/tabpfn_weights.json")


def tabpfn_source_hashes():
    return {p: file_hash(PROJECT_ROOT / p) for p in TABPFN_FILES}


def tabpfn_spec(seed):
    record = {
        "schema_version": "tabpfn-registry.g6l.v1",
        "method_id": "TabPFN",
        "method_version": TABPFN_VERSION,
        "implementation": "Prior Labs TabPFN v2 pretrained regression context model",
        "package_version": "2.1.3",
        "requires_fit": True,
        "status": "registered",
        "feature_count": 114,
        "multi_output": "two_independent_context_models",
        "seed": seed,
        "target_transform": "delta_from_current",
        "target_scaling": "train-only standard delta",
        "preprocessing": "train-only imputation, stable scaling, fixed one-hot categories and PCA8",
        "preset_parameters": dict(TABPFN_PARAMETERS),
        "sample_weight": "unsupported",
        "capabilities": {"sample_weight": False, "uncertainty": "raw_quantiles"},
        "uncertainty": "pretrained conditional marginal quantiles",
        "dependencies": {},
        "isolated_dependencies": {"tabpfn": "2.1.3", "scikit-learn": "1.6.1", "torch": "2.8.0"},
        "weight_sha256": WEIGHT_SHA256,
        "weight_revision": "4972a65a1b30806315c6f92499959ffbfc69a673",
        "attribution": "Built with PriorLabs-TabPFN",
        "license": "Prior Labs License 1.1",
        "pretraining_scope": (
            "External pretrained prior; training data provenance is limited to the published model card."
        ),
        "targets": [{"name": "cu", "unit": "g/L"}, {"name": "as", "unit": "mg/L"}],
        "device": "cpu",
        "network_access": "disabled in inference worker",
        "automatic_promotion": False,
        "causal_control": False,
        "forecast_use": "research_comparison",
        "optimization_proxy_approval": "requires_independent_qualification",
    }
    record["content_hash"] = digest(record)
    return record
