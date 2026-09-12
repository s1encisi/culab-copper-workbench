"""Fixed optional boosting models for the remaining predictive inventory."""
from copper_mvp.common import digest

SPECIALIZED_METHODS = ("CatBoost", "NGBoost", "EBM", "Cubist")
PACKAGES = {"catboost": "1.2.8", "ngboost": "0.5.6", "interpret-core": "0.7.0", "cubist": "1.2.2"}
PARAMETERS = {
    "Cubist": {"n_rules": 100, "n_committees": 5, "neighbors": None, "auto": False, "sample": None, "cv": None, "extrapolation": 0.0, "verbose": 0},
    "CatBoost": {"iterations": 200, "depth": 5, "learning_rate": 0.05, "loss_function": "RMSE",
                 "boosting_type": "Ordered", "has_time": True, "use_best_model": False,
                 "allow_writing_files": False, "thread_count": 1, "verbose": False},
    "NGBoost": {"n_estimators": 200, "learning_rate": 0.03, "natural_gradient": True,
                "minibatch_frac": 1.0, "col_sample": 1.0, "verbose": False,
                "early_stopping_rounds": None, "tol": 1e-4},
    "EBM": {"interactions": 2, "outer_bags": 1, "inner_bags": 0, "validation_size": 0,
            "early_stopping_rounds": 0, "max_rounds": 400, "learning_rate": 0.04,
            "smoothing_rounds": 100, "interaction_smoothing_rounds": 50,
            "max_bins": 64, "max_interaction_bins": 16, "min_samples_leaf": 10,
            "max_leaves": 3, "n_jobs": 1},
}


def specialized_spec(name, seed):
    package = {"CatBoost": "catboost", "NGBoost": "ngboost", "EBM": "interpret-core", "Cubist": "cubist"}[name]
    record = {"schema_version": "specialized-model-registry.g6g.v1", "method_id": name,
        "method_version": "g6g.fixed.v2", "implementation": {"CatBoost": "CatBoostRegressor",
        "NGBoost": "NGBRegressor Normal LogScore natural gradient", "EBM": "ExplainableBoostingRegressor", "Cubist": "Cubist rule-based linear regression"}[name],
        "package_version": PACKAGES[package], "dependencies": {package: PACKAGES[package]}, "requires_fit": True,
        "status": "registered", "feature_count": 114, "multi_output": "two_independent_estimators",
        "target_transform": "delta_from_current", "target_scaling": "train_only_standard_delta",
        "preprocessing": "native_numeric_missing_and_explicit_nominal_modes" if name in ("CatBoost", "EBM") else "train_median_missing_indicators_stable_scale_and_fixed_one_hot",
        "sample_weight": "unsupported" if name == "Cubist" else "native", "internal_validation": "disabled", "training_order": "input_order_used; comparison_must_verify_chronology" if name == "CatBoost" else "training_rows_only",
        "uncertainty": "uncalibrated_normal_marginals" if name == "NGBoost" else "unsupported",
        "targets": [{"name": "cu", "unit": "g/L"}, {"name": "as", "unit": "mg/L"}],
        "capabilities": {"sample_weight": name != "Cubist", "uncertainty": "marginal_std" if name == "NGBoost" else "unsupported"},
        "preset_parameters": dict(PARAMETERS[name]), "seed": seed, "causal_control": False,
        "forecast_use": "research_comparison", "automatic_promotion": False,
        "optimization_proxy_approval": "not_granted_by_model_registration"}
    record["content_hash"] = digest(record)
    return record
