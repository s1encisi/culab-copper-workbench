"""Five statistical methods from the G6 design, with event-index semantics."""
from copper_mvp.common import digest
from copper_mvp.statistical_runtime import activate_statistical_path
from copy import deepcopy

STATISTICAL_METHODS=("LocalLinearKernel","PSplineGAM","SARIMAX","VAR","ETS")
SEQUENCE_METHODS=("SARIMAX","VAR","ETS")
DESIGN_IDS={"LocalLinearKernel":"M14","PSplineGAM":"M23","SARIMAX":"M30","VAR":"M31","ETS":"M32"}
IMPLEMENTATIONS={"LocalLinearKernel":"statsmodels.nonparametric.kernel_regression.KernelReg",
 "PSplineGAM":"statsmodels.OLS_with_additive_penalized_B_spline_basis",
 "SARIMAX":"statsmodels.tsa.statespace.sarimax.SARIMAX",
 "VAR":"statsmodels.tsa.vector_ar.var_model.VAR",
 "ETS":"statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing"}
PARAMETERS={"LocalLinearKernel":{"reg_type":"ll","bandwidth":[.5,.5],"input_columns":[0,1]},
 "PSplineGAM":{"n_knots":5,"degree":3,"alpha":1.,"extrapolation":"linear","input_columns":[0,1]},
 "SARIMAX":{"order":[1,0,1],"trend":"n","exog_intercept":True,"max_iter":200,"enforce_stationarity":True,"enforce_invertibility":True,
             "exog_columns":["stage3_current_a__t_minus_0h","stage4_current_a__t_minus_0h"],"exog_alignment":"previous-origin values for observed response; current-origin values for next forecast"},
 "VAR":{"lags":2,"trend":"c","missing":"causal_forward_fill_preserving_raw_event_positions"},
 "ETS":{"trend":True,"damped_trend":True,"seasonal":None,"initialization_method":"estimated","concentrate_scale":False,"max_iter":200}}


def statistical_spec(method,seed):
    activate_statistical_path()
    sequence=method in SEQUENCE_METHODS
    result={"schema_version":"model-registry.g6b.v1","method_id":method,"design_id":DESIGN_IDS[method],
      "method_version":"g6b.fixed.v2","implementation":IMPLEMENTATIONS[method],"package_version":"0.14.6",
      "dependencies":{"statsmodels":"0.14.6","patsy":"1.0.1"},"requires_fit":True,"status":"registered","feature_count":114,
      "targets":[{"name":"cu","unit":"g/L"},{"name":"as","unit":"mg/L"}],
      "input_kind":"raw_event_sequence" if sequence else "current_result_pair",
      "target_transform":"level_series" if sequence else "delta_from_current",
      "target_scaling":"training_only_standardization","multi_output":"joint_VAR" if method=="VAR" else "two_independent_estimators",
      "preprocessing":"full_event_index_quality_mask_and_training_only_scale" if sequence else "training_only_scale_of_current_Cu_As",
      "preset_parameters":deepcopy(PARAMETERS[method]),"seed":seed,"stochastic":False,
      "capabilities":{"sample_weight":"unsupported","missing":"native_state_filter" if method in ("SARIMAX","ETS") else "causal_forward_fill_adapter" if sequence else "current_inputs_required",
        "incremental":"state_update_without_parameter_refit" if sequence else "unsupported",
        "uncertainty":"marginal_std" if sequence else "unsupported","uncertainty_calibration":"not_calibrated","joint_prediction_region":"unsupported"},
      "minimum_training_events":30,"forecast_use":"research_comparison","optimization_proxy_approval":"not_granted",
      "causal_control":False,"automatic_promotion":False,"sequence_time_basis":"recorded_event_steps_not_fixed_hours" if sequence else None}
    result["content_hash"]=digest(result)
    return result
