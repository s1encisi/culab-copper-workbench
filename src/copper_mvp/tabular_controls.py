"""Parameter-matched MLP controls for completed fixed neural inventory studies."""
from pathlib import Path
import json,time
import joblib
import numpy as np
import pandas as pd
from copper_mvp.common import PROJECT_ROOT,digest,file_hash,utc_now,write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_training import source_signature
from copper_mvp.classical_study import evaluate_classical_study
from copper_mvp.ensemble_evaluation import hierarchical_interval
from copper_mvp.tabular_models import TabularModel
from copper_mvp.tabular_registry import TABULAR_METHODS,tabular_source_hashes


def run_matched_controls(data,run_root,study_id,request_key,progress=print):
    run_root=Path(run_root);source=run_root/"model_inventory_studies"/study_id
    state=json.loads((source/"state.json").read_text())
    assert state["status"]=="completed"
    original=json.loads((source/"protocol.json").read_text())
    base=pd.read_csv(source/"predictions.csv")
    seeds=original["seeds"]
    methods=[m["method_id"] for m in original["methods"] if m["method_id"] in TABULAR_METHODS]
    assert set(methods)==set(TABULAR_METHODS)
    assert source_signature(data)==original["source"]
    root=run_root/"neural_controls"/digest(request_key)[:32]
    root.mkdir(parents=True,exist_ok=False)
    controls={method:"MLP_for_"+method for method in methods}
    expected=dict(original["expected_method_seeds"])
    expected.update({name:seeds for name in controls.values()})
    protocol={**original,"schema_version":"neural-mlp-controls.g6j.v1",
        "request_key":request_key,"expected_method_seeds":expected,"controls":controls,
        "source_study_id":study_id,"source_predictions_sha256":file_hash(source/"predictions.csv"),
        "source_evaluation_sha256":file_hash(source/"evaluation.json"),
        "control_code_hashes":tabular_source_hashes(),"new_methods_added":0,
        "matching":"same preprocessing, target scaling, AdamW, learning rate, epochs, batch sequence and weights; parameter difference <0.1%"}
    write_json(root/"protocol.json",protocol)
    status={"status":"running","started_at":utc_now(),"artifacts":[]}
    write_json(root/"state.json",status)
    ledger=DataService(data).labels
    rows=[];resources=[]
    scopes=[]
    for fold in sorted(set(data.fold_for.values())):
        frame=data.cv[data.cv.fold_id.eq(fold)]
        ids=data.training_ids(fold)
        valid=frame[frame.fold_role.eq("VALIDATION")].origin_event_id.tolist()
        scopes.append((fold,ids,valid,source_time(frame.fold_fit_cutoff_at.iloc[0])))
    scopes.append(("DEVELOPMENT",data.training_ids(None),[],source_time(data.frame.decision_at.max())))
    started=time.perf_counter()
    try:
        for seed in seeds:
            comparison=next(r["comparison_id"] for r in state["comparisons"] if r["seed"]==seed)
            candidate_manifest=json.loads((run_root/"model_comparisons"/comparison/"training_manifest.json").read_text())
            for fold,ids,valid,cutoff in scopes:
                labels=ledger.latest(cutoff)
                assert all(labels[(e,t)].quality_eligible and labels[(e,t)].value is not None for e in ids for t in ("cu","as"))
                X=data.X.loc[ids].to_numpy(float)
                y=np.array([[labels[(e,t)].value for t in ("cu","as")] for e in ids])
                for method in methods:
                    progress({"seed":seed,"fold":fold,"control":controls[method]})
                    model=TabularModel(method,seed,reference=True)
                    model.fit(X,y,max_wall_seconds=1800)
                    candidate=next(a for a in candidate_manifest["artifacts"] if a["fold_id"]==fold and a["method_id"]==method)
                    assert model.fit_metadata["optimizer_updates"]==candidate["fit_metadata"]["optimizer_updates"]
                    assert model.fit_metadata["parameter_budget"]==candidate["fit_metadata"]["parameters"]
                    assert model.fit_metadata["parameter_budget_relative_difference"]<.001
                    path=root/"artifacts"/str(seed)/fold/(controls[method]+".joblib")
                    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path,compress=3)
                    status["artifacts"].append({"seed":seed,"fold":fold,"method":controls[method],
                        "path":path.relative_to(root).as_posix(),"sha256":file_hash(path),
                        "train_event_hash":digest(ids),"fit_cutoff":cutoff.isoformat()})
                    resources.append({"seed":seed,"fold":fold,"method":controls[method],**model.fit_metadata})
                    if valid:
                        prediction=model.predict(data.X.loc[valid].to_numpy(float))
                        for e,value in zip(valid,prediction):
                            rows.append({"event_id":e,"fold_id":fold,"method_id":controls[method],"seed":seed,
                                         "cu":float(value[0]),"as":float(value[1]),"status":"completed"})
                    write_json(root/"state.json",status)
        assert source_signature(data)==original["source"]
        assert tabular_source_hashes()==protocol["control_code_hashes"]
        table=pd.concat([base,pd.DataFrame(rows)],ignore_index=True)
        table.to_csv(root/"predictions.csv",index=False)
        result=evaluate_classical_study(data,table,protocol)
        events=sorted(data.fold_for,key=lambda e:(data.row(e).decision_at,e))
        labels=ledger.latest(protocol["evaluation_as_of"])
        dates=[source_time(data.row(e).decision_at).isoformat() for e in events]
        paired={}
        for method,control in controls.items():
            for target in ("cu","as"):
                truth=np.array([labels[(e,target)].value for e in events])
                differences=[]
                for seed in seeds:
                    main=table[(table.method_id==method)&(table.seed==seed)].set_index("event_id").loc[events,target].to_numpy()
                    reference=table[(table.method_id==control)&(table.seed==seed)].set_index("event_id").loc[events,target].to_numpy()
                    differences.append(abs(main-truth)-abs(reference-truth))
                differences=np.array(differences)
                paired[method+":"+target]={"candidate_minus_mlp_mae":float(differences.mean()),
                                          "ci95":hierarchical_interval(differences,dates)}
        result.update(schema_version="neural-control-evaluation.g6j.v1",new_method_count=0,
            candidate_vs_matched_mlp=paired,resources=resources,elapsed_seconds=time.perf_counter()-started,
            source_study_id=study_id,protocol_sha256=file_hash(root/"protocol.json"),
            predictions_sha256=file_hash(root/"predictions.csv"))
        write_json(root/"evaluation.json",result)
        status.update(status=result["status"],finished_at=utc_now(),evaluation_sha256=file_hash(root/"evaluation.json"))
        write_json(root/"state.json",status)
        return root,result
    except Exception as exc:
        status.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)[:400]})
        write_json(root/"state.json",status)
        raise
