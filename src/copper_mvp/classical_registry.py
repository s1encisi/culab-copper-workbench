"""Distinct classical methods from the G6 inventory; fixed first-pass settings."""
from __future__ import annotations
import sklearn
from copy import deepcopy
from copper_mvp.common import digest

CLASSICAL_METHODS=("BayesianRidge","ARD","RANSAC","TheilSen","Quantile","MultiTaskElasticNet",
                   "KernelRidge","SVR","KNN","GaussianProcess","DecisionTree","RandomForest","AdaBoost","MLP")
STOCHASTIC_METHODS=("RANSAC","TheilSen","DecisionTree","RandomForest","AdaBoost","MLP")
DESIGN_IDS={"BayesianRidge":"M02","ARD":"M03","RANSAC":"M05","TheilSen":"M06","Quantile":"M07",
            "MultiTaskElasticNet":"M08","KernelRidge":"M10","SVR":"M11","KNN":"M12","GaussianProcess":"M13",
            "DecisionTree":"M15","RandomForest":"M16","AdaBoost":"M17","MLP":"M25"}
CLASSES={"BayesianRidge":"sklearn.linear_model.BayesianRidge","ARD":"sklearn.linear_model.ARDRegression",
 "RANSAC":"sklearn.linear_model.RANSACRegressor","TheilSen":"sklearn.linear_model.TheilSenRegressor",
 "Quantile":"sklearn.linear_model.QuantileRegressor","MultiTaskElasticNet":"sklearn.linear_model.MultiTaskElasticNet",
 "KernelRidge":"sklearn.kernel_ridge.KernelRidge","SVR":"sklearn.svm.SVR","KNN":"sklearn.neighbors.KNeighborsRegressor",
 "GaussianProcess":"sklearn.gaussian_process.GaussianProcessRegressor","DecisionTree":"sklearn.tree.DecisionTreeRegressor",
 "RandomForest":"sklearn.ensemble.RandomForestRegressor","AdaBoost":"sklearn.ensemble.AdaBoostRegressor",
 "MLP":"sklearn.neural_network.MLPRegressor"}
WEIGHTED={"BayesianRidge","RANSAC","Quantile","KernelRidge","SVR","DecisionTree","RandomForest","AdaBoost","MLP"}
JOINT={"MultiTaskElasticNet","KernelRidge","KNN","GaussianProcess","DecisionTree","RandomForest","MLP"}
REDUCTION={"RANSAC":5,"TheilSen":5,"GaussianProcess":8}
PARAMETERS={
 "BayesianRidge":{"max_iter":500,"tol":1e-5},
 "ARD":{"max_iter":500,"tol":1e-5,"threshold_lambda":10000.},
 "RANSAC":{"estimator":"Ridge(alpha=1.0)","min_samples":6,"residual_threshold":.5,"max_trials":100,"stop_probability":.99},
 "TheilSen":{"max_subpopulation":1000,"max_iter":300,"tol":1e-3,"n_jobs":1},
 "Quantile":{"quantiles":[.1,.5,.9],"alpha":.01,"solver":"highs"},
 "MultiTaskElasticNet":{"alpha":.01,"l1_ratio":.5,"max_iter":20000,"tol":1e-6,"selection":"cyclic"},
 "KernelRidge":{"alpha":1.,"kernel":"rbf","gamma":.01},
 "SVR":{"C":10.,"epsilon":.1,"kernel":"rbf","gamma":"scale"},
 "KNN":{"n_neighbors":15,"weights":"distance","p":2},
 "GaussianProcess":{"kernel":"1.0 * RBF(length_scale=1.0), fixed","alpha":.05,"optimizer":None,"normalize_y":False},
 "DecisionTree":{"max_depth":8,"min_samples_leaf":5},
 "RandomForest":{"n_estimators":200,"max_depth":12,"min_samples_leaf":3,"max_features":.8,"bootstrap":True,"n_jobs":1},
 "AdaBoost":{"estimator":"DecisionTreeRegressor(max_depth=3,min_samples_leaf=5)","n_estimators":100,"learning_rate":.05,"loss":"linear"},
 "MLP":{"hidden_layer_sizes":[32,16],"solver":"lbfgs","alpha":.01,"max_iter":1000,"max_fun":20000,"tol":1e-5,"early_stopping":False},
}


def classical_spec(method_id,seed):
    uncertainty="marginal_std" if method_id in ("BayesianRidge","ARD","GaussianProcess") else "raw_quantiles" if method_id=="Quantile" else "unsupported"
    spec={"schema_version":"model-registry.g6a.v1","method_id":method_id,"design_id":DESIGN_IDS[method_id],
        "method_version":"g6a.fixed.v1","implementation":CLASSES[method_id],"package_version":sklearn.__version__,
        "requires_fit":True,"status":"registered","feature_count":114,
        "targets":[{"name":"cu","unit":"g/L"},{"name":"as","unit":"mg/L"}],
        "target_transform":"delta_from_current","target_scaling":"train_only_standard_delta",
        "multi_output":"native_independent_outputs_shared_kernel" if method_id=="GaussianProcess" else "native_joint" if method_id in JOINT else "two_independent_estimators",
        "preprocessing":"train_median_with_missing_indicators_stable_numeric_scale_fixed_mode_one_hot",
        "preprocessing_version":"g6a.numeric64eps.v1","pca_components":REDUCTION.get(method_id),
        "numerical_constant_policy":"Training variance <= (64*float64_eps*max(1,max_abs_training_column))^2 is inactive",
        "preset_parameters":deepcopy(PARAMETERS[method_id]),"seed":seed,"stochastic":method_id in STOCHASTIC_METHODS,
        "capabilities":{"missing":"adapter","categorical":"adapter","sample_weight":"adapter" if method_id in WEIGHTED else "unsupported",
            "sample_weight_scope":"Estimator loss only; preprocessing and target scaling fitted without weights",
            "incremental":"unsupported","uncertainty":uncertainty,"uncertainty_calibration":"not_calibrated",
            "joint_prediction_region":"unsupported",
            "gaussian_process_std":"observation_std_including_fixed_alpha_noise" if method_id=="GaussianProcess" else None},
        "minimum_training_events":16 if method_id=="KNN" else 10,"forecast_use":"research_comparison","optimization_proxy_approval":"not_granted_by_registration",
        "causal_control":False,"automatic_promotion":False,"parameter_policy":"predeclared_fixed_configuration_no_outer_label_tuning"}
    spec["content_hash"]=digest(spec)
    return spec
