"""Explicit CPU configurations for three structurally distinct tabular neural models."""
from copper_mvp.common import PROJECT_ROOT,digest,file_hash
TABULAR_METHODS=("TabNet","FTTransformer","NODE")
TABULAR_VERSION="g6j.tabular.v1"
TRAINING={"epochs":60,"batch_size":128,"learning_rate":0.001,"weight_decay":0.0001,"gradient_clip":5.0}
ARCHITECTURES={
    "TabNet":{"n_d":8,"n_a":8,"n_steps":3,"cat_emb_dim":2,"n_shared":2,"n_independent":2,
              "virtual_batch_size":64,"gamma":1.3,"lambda_sparse":0.001},
    "FTTransformer":{"n_blocks":2,"d_block":32,"attention_n_heads":4,"attention_dropout":0.1,
                     "ffn_d_hidden":64,"ffn_dropout":0.1,"residual_dropout":0.0},
    "NODE":{"layers":2,"trees_per_layer":32,"depth":4,"tree_dim":2,
             "selector":"sparsemax","split":"sparsemoid","threshold_initialization":"train_quantiles"},
}
TABULAR_PACKAGES={"pytorch-tabnet":"4.1.0","rtdl-revisiting-models":"0.0.2","entmax":"1.3"}
TABULAR_SOURCE_FILES=tuple("src/copper_mvp/"+p for p in (
    "tabular_registry.py","tabular_runtime.py","tabular_inputs.py","tabular_networks.py",
    "tabular_models.py","node_network.py","neural_runtime.py"))+("requirements-tabular-neural.txt",)


def tabular_source_hashes():
    return {p:file_hash(PROJECT_ROOT/p) for p in TABULAR_SOURCE_FILES}


def tabular_spec(method,seed):
    record={"schema_version":"tabular-registry.g6j.v1","method_id":method,"method_version":TABULAR_VERSION,
        "implementation":{"TabNet":"native pytorch-tabnet attentive network","FTTransformer":"author rtdl FTTransformer",
                          "NODE":"dense differentiable oblivious tree ensemble with sparsemax"}[method],
        "package_version":{"TabNet":"4.1.0","FTTransformer":"0.0.2","NODE":"g6j.node.v1"}[method],
        "dependencies":{"torch":"2.8.0",**TABULAR_PACKAGES},"requires_fit":True,"status":"registered",
        "feature_count":114,"multi_output":"joint_two_target_head","seed":seed,
        "target_transform":"delta_from_current","target_scaling":"training_only_standard_delta",
        "preprocessing":"train_median_stable_scale; all_numeric_missing_masks; explicit_mode_codes",
        "preset_parameters":{"training":dict(TRAINING),"architecture":dict(ARCHITECTURES[method])},
        "sample_weight":"weighted_two_target_MSE","capabilities":{"sample_weight":True,"uncertainty":"unsupported"},
        "uncertainty":"unsupported","internal_validation":"disabled","early_stopping":False,
        "targets":[{"name":"cu","unit":"g/L"},{"name":"as","unit":"mg/L"}],
        "device":"cpu","native_threads":1,"forecast_use":"research_comparison",
        "automatic_promotion":False,"causal_control":False,"optimization_proxy_approval":"requires_independent_qualification"}
    record["content_hash"]=digest(record)
    return record
