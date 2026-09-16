"""Assemble an inventory evaluation from verified completed comparison runs."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

if __name__=="__main__":
    import numpy as np,pandas as pd
    from copper_mvp.common import PROJECT_ROOT,digest,file_hash,write_json,utc_now
    from copper_mvp.data import DataRepository
    from copper_mvp.model_comparisons import ComparisonService
    from copper_mvp.model_evaluation import read_training
    from copper_mvp.model_training import source_signature
    from copper_mvp.classical_study import evaluate_classical_study
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-key",required=True)
    parser.add_argument("--comparison-id",action="append",required=True)
    parser.add_argument("--reference-comparison-id",required=True)
    parser.add_argument("--run-root",type=Path,default=Path("runs/mvp"))
    args=parser.parse_args()
    run_root=args.run_root.resolve()
    assert run_root.is_relative_to((PROJECT_ROOT/"runs").resolve())
    data=DataRepository();service=ComparisonService(run_root/"model_comparisons",data)
    frames=[];resources=[];uncertainty=[];sources=[];seeds=[];specification=None
    for identifier in args.comparison_id:
        record=service.get(identifier);assert record["status"]=="completed" and record["result"]["status"]=="completed"
        folder=service.directory(identifier);manifest,protocol=read_training(folder)
        assert manifest["source"]==source_signature(data)
        normal=[{k:v for k,v in s.items() if k not in ("seed","content_hash")} for s in protocol["methods"]]
        if specification is None:
            specification=normal;first=protocol
        else:
            assert normal==specification,"Comparison configurations differ"
            assert protocol["code_hashes"]==first["code_hashes"],"Training implementations differ"
        seed=protocol["request"]["seed"];assert seed not in seeds;seeds.append(seed)
        frame=pd.read_csv(folder/"oof_predictions.csv");frame["seed"]=seed;frames.append(frame)
        resources.extend({**r,"seed":seed,"comparison_id":identifier} for r in record["result"]["resources"])
        uncertainty.extend({**r,"seed":seed,"comparison_id":identifier} for r in record["result"].get("uncertainty_metrics",[]))
        sources.append({"seed":seed,"comparison_id":identifier,"status":"completed",
            "manifest_sha256":file_hash(folder/"training_manifest.json"),"oof_sha256":manifest["oof_sha256"],
            "evaluation_sha256":file_hash(folder/"evaluation.json")})
    order=np.argsort(seeds);seeds=[seeds[i] for i in order];sources=[sources[i] for i in order]
    methods=[m for m in first["request"]["methods"] if m!="Persistence"]
    reference=service.get(args.reference_comparison_id);assert reference["status"]=="completed"
    folder=service.directory(args.reference_comparison_id);manifest,reference_protocol=read_training(folder)
    assert manifest["source"]==source_signature(data)
    reference_seed=reference_protocol["request"]["seed"]
    baseline_methods=[m for m in reference_protocol["request"]["methods"] if m not in ("Persistence",*methods)]
    old=pd.read_csv(folder/"oof_predictions.csv");old=old[old.method_id.isin(baseline_methods)].copy();old["seed"]=reference_seed
    frames.append(old);table=pd.concat(frames,ignore_index=True)
    table=table[~(table.method_id.eq("Persistence")&table.seed.ne(seeds[0]))]
    expected={"Persistence":[seeds[0]],**{m:seeds for m in methods},**{m:[reference_seed] for m in baseline_methods}}
    root=run_root/"model_inventory_studies"/digest(args.request_key)[:32];root.mkdir(parents=True,exist_ok=False)
    source_hashes={(key if "/" in key else "src/copper_mvp/"+key):value for key,value in first["code_hashes"].items()}
    for path,expected_hash in source_hashes.items():
        assert file_hash(PROJECT_ROOT/path)==expected_hash,path
        destination=root/"executed_source"/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((PROJECT_ROOT/path).read_bytes())
    protocol={"schema_version":"recovered-inventory-study.v1","request_key":args.request_key,"seeds":seeds,
        "methods":[m for m in first["methods"] if m["method_id"]!="Persistence"],
        "expected_method_seeds":expected,"evaluation_as_of":first["evaluation_as_of"],"source":manifest["source"],
        "reference_comparison_id":args.reference_comparison_id,"source_comparisons":sources,
        "code_hashes":source_hashes,"recovery_note":"Existing completed fits and predictions are reused after hash/configuration verification; no validation-based selection."}
    write_json(root/"protocol.json",protocol);table.to_csv(root/"predictions.csv",index=False)
    result=evaluate_classical_study(data,table,protocol)
    result.update(new_method_count=len(methods),source_comparisons=sources,resources=resources,
        uncertainty_metrics=uncertainty,source_reference=args.reference_comparison_id,
        protocol_sha256=file_hash(root/"protocol.json"),predictions_sha256=file_hash(root/"predictions.csv"),
        recovery_assembler_sha256=file_hash(Path(__file__)),created_at=utc_now())
    write_json(root/"evaluation.json",result)
    write_json(root/"state.json",{"status":result["status"],"id":root.name,"comparisons":sources,
                                 "evaluation_sha256":file_hash(root/"evaluation.json")})
    print(json.dumps({"status":result["status"],"study_id":root.name,"seeds":seeds,"common_events":result["common_events"]}))
