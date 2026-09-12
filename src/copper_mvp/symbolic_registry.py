"""Fixed symbolic regression protocol on an explicit process-variable subset."""
from copper_mvp.common import PROJECT_ROOT,digest,file_hash

SYMBOLIC_METHODS=("SymbolicRegression",)
SYMBOLIC_VERSION="g6i.symbolic.v1"
SYMBOLIC_FEATURE_INDICES=(0,1,78,82,90,94,98,106)
SYMBOLIC_FEATURE_ROLES=("current_cu","current_as","stage3_temperature","stage3_current",
                        "stage3_flow","stage4_temperature","stage4_current","stage4_flow")
SYMBOLIC_PARAMETERS={
    "niterations":40,"populations":8,"population_size":27,"ncycles_per_iteration":100,
    "maxsize":15,"maxdepth":6,"max_evals":100000,
    "parsimony":0.001,"model_selection":"best","precision":64,
    "parallelism":"serial","deterministic":True,"batching":False,
    "turbo":False,"bumper":False,"progress":False,"verbosity":0,
    "binary_operators":["+","-","*","safe_div(x, y) = x / (abs(y) + 1.0)"],
    "unary_operators":["tanh"],
    "constraints":{"*":(5,5),"safe_div":(-1,4)},
}
SYMBOLIC_SOURCE_FILES=tuple("src/copper_mvp/"+name for name in (
    "symbolic_registry.py","symbolic_runtime.py","symbolic_models.py",
    "symbolic_sampling.py","symbolic_expression.py"))+(
    "scripts/run_symbolic_fit.py","requirements-symbolic-models.txt")


def symbolic_source_hashes():
    return {p:file_hash(PROJECT_ROOT/p) for p in SYMBOLIC_SOURCE_FILES}


def symbolic_spec(seed):
    record={
        "schema_version":"symbolic-registry.g6i.v1","method_id":"SymbolicRegression",
        "method_version":SYMBOLIC_VERSION,"implementation":"PySR SymbolicRegression.jl evolutionary expression search",
        "package_version":"1.5.9","requires_fit":True,"status":"registered","feature_count":114,
        "selected_feature_indices":list(SYMBOLIC_FEATURE_INDICES),
        "selected_feature_roles":list(SYMBOLIC_FEATURE_ROLES),
        "multi_output":"two_independent_expression_searches","seed":seed,
        "target_transform":"delta_from_current","target_scaling":"training_only_standard_delta",
        "preprocessing":"eight_declared_process_inputs; training_median_and_stable_scale",
        "preset_parameters":dict(SYMBOLIC_PARAMETERS),
        "training_dependencies":{"pysr":"1.5.9","juliacall":"0.9.26","julia":"1.11.9","SymbolicRegression.jl":"1.11.3"},
        "dependencies":{},"sample_weight":"unsupported",
        "capabilities":{"sample_weight":False,"uncertainty":"unsupported"},
        "uncertainty":"unsupported","internal_validation":"disabled",
        "targets":[{"name":"cu","unit":"g/L"},{"name":"as","unit":"mg/L"}],
        "dimensional_policy":"All searched variables and responses are dimensionless; physical units restored by stored affine transforms.",
        "operator_domain":"safe_div denominator is abs(y)+1; no division poles or log/sqrt domains",
        "inference":"validated expression tree evaluated with NumPy; no Julia serving dependency",
        "forecast_use":"research_comparison","automatic_promotion":False,"causal_control":False,
        "optimization_proxy_approval":"requires_independent_qualification"}
    record["content_hash"]=digest(record)
    return record
