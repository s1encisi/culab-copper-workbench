"""BART's fixed research configuration and explicit posterior contract."""

from copper_mvp.common import PROJECT_ROOT, digest, file_hash

BART_SOURCE_FILES = tuple(
    "src/copper_mvp/" + name
    for name in ("bart_registry.py", "bart_models.py", "bart_runtime.py", "bart_sampling.py", "bart_trees.py")
) + ("scripts/run_bart_fit.py", "requirements-bart-models.txt")


def bart_source_hashes():
    return {path: file_hash(PROJECT_ROOT / path) for path in BART_SOURCE_FILES}


BART_METHODS = ("BART",)
BART_VERSION = "g6h.bart.v3"
BART_PARAMETERS = {
    "components": 8,
    "trees": 20,
    "tune": 400,
    "draws": 600,
    "chains": 4,
    "pytensor_mode": "NUMBA",
    "particles": 10,
    "batch_fraction": 0.25,
    "target_accept": 0.9,
    "sigma_prior_scale": 1.0,
    "alpha": 0.95,
    "beta": 2.0,
    "maximum_rhat": 1.05,
    "minimum_ess": 100,
    "max_fit_seconds": 3600,
}
BART_PACKAGES = {
    "pymc-bart": "0.7.0",
    "pymc": "5.16.2",
    "pytensor": "2.25.5",
    "arviz": "0.20.0",
    "numba": "0.61.2",
    "llvmlite": "0.44.0",
}


def bart_spec(seed):
    spec = {
        "schema_version": "bart-registry.g6h.v1",
        "method_id": "BART",
        "method_version": BART_VERSION,
        "implementation": "PyMC-BART PGBART posterior tree ensembles",
        "package_version": "0.7.0",
        "training_dependencies": dict(BART_PACKAGES),
        "dependencies": {"numba": "0.61.2", "llvmlite": "0.44.0"},
        "requires_fit": True,
        "status": "registered",
        "feature_count": 114,
        "multi_output": "two_independent_posterior_models",
        "target_transform": "delta_from_current",
        "target_scaling": "training_only_standard_delta",
        "seed": seed,
        "preprocessing": "training_only_impute_stable_scale_fixed_one_hot_PCA8",
        "preset_parameters": dict(BART_PARAMETERS),
        "sample_weight": "unsupported",
        "capabilities": {"sample_weight": False, "uncertainty": "raw_quantiles"},
        "uncertainty": "posterior_normal_mixture_quantiles",
        "internal_validation": "disabled",
        "targets": [{"name": "cu", "unit": "g/L"}, {"name": "as", "unit": "mg/L"}],
        "inference": "exported_numeric_trees; training runtime not loaded during prediction",
        "forecast_use": "research_comparison",
        "automatic_promotion": False,
        "causal_control": False,
        "optimization_proxy_approval": "requires_independent_qualification",
    }
    spec["content_hash"] = digest(spec)
    return spec
