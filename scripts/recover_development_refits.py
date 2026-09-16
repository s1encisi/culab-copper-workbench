"""Recover a comparison with complete OOF predictions but missing development refits."""
from pathlib import Path
import argparse,json,os,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["LANGSMITH_TRACING"]=os.environ["LANGCHAIN_TRACING_V2"]="false"
if __name__=="__main__":
    import joblib,numpy as np
    from copper_mvp.common import PROJECT_ROOT,digest,file_hash,utc_now,write_json
    from copper_mvp.data import DataRepository
    from copper_mvp.data_service import DataService
    from copper_mvp.data_contracts import source_time
    from copper_mvp.model_registry import method_spec
    from copper_mvp.model_adapters import RegisteredModel
    from copper_mvp.model_comparisons import process_alive
    from copper_mvp.model_training import source_signature
    from copper_mvp.model_evaluation import read_training,evaluate_comparison
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-id",required=True)
    parser.add_argument("--request-key",required=True)
    args=parser.parse_args()
    assert len(args.source_id)==32 and all(c in "0123456789abcdef" for c in args.source_id)
    base=PROJECT_ROOT/"runs/mvp/model_comparisons"
    source=base/args.source_id;state=json.loads((source/"state.json").read_text())
    assert not process_alive(state.get("owner_pid"))
    manifest,protocol=read_training(source)
    data=DataRepository();assert source_signature(data)==manifest["source"]
    methods=protocol["request"]["methods"];folds=[f["fold_id"] for f in protocol["folds"] if f["fold_id"]!="DEVELOPMENT"]
    completed={(a["fold_id"],a["method_id"]) for a in manifest["artifacts"] if a["status"]=="completed"}
    assert {(f,m) for f in folds for m in methods}.issubset(completed)
    for artifact in manifest["artifacts"]:
        if artifact["status"]=="completed":assert file_hash(source/artifact["path"])==artifact["sha256"]
    identifier=digest(args.request_key)[:32];root=base/identifier
    assert not root.exists()
    shutil.copytree(source,root)
    protocol["request"]["request_key"]=args.request_key
    protocol["recovery"]={"source_comparison":args.source_id,"source_manifest_sha256":file_hash(source/"training_manifest.json"),
                          "source_oof_sha256":manifest["oof_sha256"],"script_sha256":file_hash(Path(__file__)),
                          "scope":"reuse all verified OOF artifacts; fit only missing DEVELOPMENT models",
                          "unrecorded_interrupted_compute_cost":"unknown"}
    write_json(root/"protocol.json",protocol)
    state.update(run_id=identifier,status="running",owner_pid=os.getpid(),created_at=utc_now(),
                 request=protocol["request"],fingerprint=digest(protocol["recovery"]))
    write_json(root/"state.json",state)
    ids=data.training_ids(None);cutoff=source_time(data.frame.decision_at.max())
    labels=DataService(data).labels.latest(cutoff)
    assert all(labels[(e,t)].quality_eligible and labels[(e,t)].value is not None for e in ids for t in ("cu","as"))
    X=data.X.loc[ids].to_numpy(float);y=np.array([[labels[(e,t)].value for t in ("cu","as")] for e in ids])
    started=time.perf_counter();old_elapsed=manifest["elapsed_ms"]
    for method in methods:
        if ("DEVELOPMENT",method) in completed:continue
        print(json.dumps({"recovery":identifier,"method":method}),flush=True)
        model=RegisteredModel(method,protocol["request"]["seed"])
        clock=time.perf_counter()
        remaining=protocol["request"]["max_wall_seconds"]-old_elapsed/1000-(time.perf_counter()-started)
        assert remaining>0
        model.fit(X,y,max_wall_seconds=remaining)
        path=root/"artifacts/DEVELOPMENT"/method/"model.joblib";path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,path,compress=3)
        entry={"fold_id":"DEVELOPMENT","method_id":method,"fit_cutoff_at":cutoff.isoformat(),
               "scope":"development_analysis","train_count":len(ids),"train_event_hash":digest(ids),
               "requires_fit":True,"spec_hash":model.spec["content_hash"],"fit_metadata":model.fit_metadata,
               "path":path.relative_to(root).as_posix(),"sha256":file_hash(path),"status":"completed"}
        manifest["artifacts"]=[a for a in manifest["artifacts"] if (a["fold_id"],a["method_id"])!=("DEVELOPMENT",method)]+[entry]
        manifest["timings"]=[t for t in manifest["timings"] if (t["fold_id"],t["method_id"])!=("DEVELOPMENT",method)]
        manifest["timings"].append({"fold_id":"DEVELOPMENT","method_id":method,"train_events":len(ids),
           "status":"completed","fit_ms":(time.perf_counter()-clock)*1000,"batch_predict_ms":0,
           "single_prediction_ms":[],"warnings":model.fit_warnings,"retries":0,"cache_hits":0})
        manifest.update(protocol_sha256=file_hash(root/"protocol.json"),elapsed_ms=old_elapsed+(time.perf_counter()-started)*1000)
        write_json(root/"training_manifest.json",manifest)
    assert source_signature(data)==manifest["source"]
    manifest["status"]="completed";manifest["recovery"]=protocol["recovery"]
    write_json(root/"training_manifest.json",manifest)
    result=evaluate_comparison(data,root);assert result["status"]=="completed"
    state.update(status="completed",finished_at=utc_now(),evaluation_status=result["status"],
                 evaluation_sha256=file_hash(root/"evaluation.json"),progress={"phase":"completed"})
    write_json(root/"state.json",state)
    print(json.dumps({"status":"completed","comparison_id":identifier,"reused_oof_sha256":manifest["oof_sha256"]}),flush=True)
