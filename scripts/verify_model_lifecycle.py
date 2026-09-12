"""Verify real artifact registration and historical runtime shadow gates."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="1"
os.environ["LANGSMITH_TRACING"]="false"
os.environ["LANGCHAIN_TRACING_V2"]="false"

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,default=Path("runs/mvp"))
    parser.add_argument("--model-comparison-id",required=True)
    parser.add_argument("--ensemble-study-id",required=True)
    args=parser.parse_args()
    from copper_mvp.access import AccessControl
    from copper_mvp.common import WorkbenchError,write_json
    from copper_mvp.data import DataRepository
    from copper_mvp.storage import RunStore
    from copper_mvp.research_store import ResearchStore
    from copper_mvp.model_lifecycle import ModelLifecycle
    from copper_mvp.release_sources import ArtifactSources
    store=RunStore(args.output)
    access=AccessControl(ResearchStore(store))
    actor=access.authenticate(key=access.owner_key_path.read_text().strip())
    sources=ArtifactSources(args.source_root,DataRepository())
    service=ModelLifecycle(store,access,sources)
    cases=[
        ("model_comparison",args.model_comparison_id,"DeltaHGB",None),
        ("ensemble_study",args.ensemble_study_id,"OOFConvex",None),
        ("ensemble_study",args.ensemble_study_id,"DynamicConvex",None),
        ("ensemble_study",args.ensemble_study_id,"Bagging_d1",20260915)]
    results=[]
    for index,(kind,source,method,seed) in enumerate(cases):
        print(json.dumps({"phase":"register","method":method}),flush=True)
        artifact=service.register(actor,{"request_key":"artifact-"+str(index),"source_kind":kind,"source_id":source,
            "method_id":method,"seed":seed,"target":"cu"})
        print(json.dumps({"phase":"shadow","method":method,"candidate_id":artifact["id"]}),flush=True)
        shadow=service.start_shadow(actor,artifact["id"],{"request_key":"shadow-"+str(index),"samples_per_fold":4})
        assert shadow["status"]=="completed",shadow.get("evidence")
        gate=service.qualification(actor,artifact["id"],shadow["id"])
        rejected=False
        try:
            service.propose(actor,{"request_key":"publish-"+str(index),"candidate_id":artifact["id"],"shadow_id":shadow["id"],
                "target":"cu","expected_version":0,"reason":"Verify benchmark gate without changing defaults"})
        except WorkbenchError as exc:
            assert exc.code=="RELEASE_NOT_QUALIFIED"
            rejected=True
        assert rejected
        item={"method":method,"candidate_id":artifact["id"],"shadow_id":shadow["id"],"shadow_status":shadow["status"],
            "samples":shadow["evidence"]["samples"],"parity_passed":shadow["evidence"]["parity_passed"],"qualification":gate,
            "release_rejected":rejected}
        results.append(item)
        write_json(args.output/"verification.json",{"status":"running","cases":results,"defaults_unchanged":True})
        print(json.dumps({"phase":"verified","method":method,"gate_failures":gate["failures"]}),flush=True)
    pointers=service.pointers(actor)
    assert pointers["cu"]["version"]==pointers["as"]["version"]==0
    report={"status":"PASSED","cases":results,"pointers":pointers,"defaults_unchanged":True,"live_model_publish":False,"paid_api_calls":0}
    write_json(args.output/"verification.json",report)
    print(json.dumps({"status":"PASSED","cases":len(results),"defaults_unchanged":True}),flush=True)
