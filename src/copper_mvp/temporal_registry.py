"""Fixed causal process-sequence models, separate from event-state forecasters."""
from copper_mvp.common import PROJECT_ROOT,digest,file_hash
TEMPORAL_METHODS=("GRU","LSTM","CausalTCN")
TEMPORAL_VERSION="g6k.temporal.v1"
TEMPORAL_TRAINING={"epochs":60,"batch_size":128,"learning_rate":0.001,"weight_decay":0.0001,"gradient_clip":5.0}
TEMPORAL_ARCHITECTURES={"GRU":{"hidden":32},"LSTM":{"hidden":26},
                       "CausalTCN":{"channels":20,"kernel_size":3,"dilations":[1,2,4],"dropout":0.1}}
TEMPORAL_FILES=tuple("src/copper_mvp/"+name for name in ("temporal_registry.py","temporal_inputs.py",
"temporal_networks.py","temporal_models.py","neural_runtime.py"))


def temporal_source_hashes():
    return {p:file_hash(PROJECT_ROOT/p) for p in TEMPORAL_FILES}


def temporal_spec(method,seed):
    record={"schema_version":"temporal-registry.g6k.v1","method_id":method,"method_version":TEMPORAL_VERSION,
        "implementation":{"GRU":"torch GRU on causal process anchors","LSTM":"torch LSTM on causal process anchors",
                          "CausalTCN":"left-padded dilated residual temporal convolutions"}[method],
        "package_version":"2.8.0","dependencies":{"torch":"2.8.0"},
        "requires_fit":True,"status":"registered","feature_count":114,"input_kind":"anchored_process_sequence",
        "multi_output":"joint_two_target_head","seed":seed,"target_transform":"delta_from_current",
        "target_scaling":"training_only_standard_delta","preprocessing":"train-only scaling of 27 signals and current results; missing, age and observation-gap channels",
        "sequence":{"steps":7,"order":"oldest_to_current","offsets_hours":[12,10,8,6,4,2,0],
                    "observation_tolerance_hours":2,"hidden_state":"reset_for_each_window","labels_in_sequence":False},
        "preset_parameters":{"training":dict(TEMPORAL_TRAINING),"architecture":dict(TEMPORAL_ARCHITECTURES[method])},
        "sample_weight":"unsupported","capabilities":{"sample_weight":False,"uncertainty":"unsupported"},
        "uncertainty":"unsupported","internal_validation":"disabled","early_stopping":False,
        "targets":[{"name":"cu","unit":"g/L"},{"name":"as","unit":"mg/L"}],
        "device":"cpu","native_threads":1,"automatic_promotion":False,"causal_control":False,
        "forecast_use":"research_comparison","optimization_proxy_approval":"requires_independent_qualification"}
    record["content_hash"]=digest(record)
    return record
