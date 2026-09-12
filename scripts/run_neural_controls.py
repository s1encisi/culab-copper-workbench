"""Run parameter-and-budget-matched MLP controls after a neural inventory study."""
from pathlib import Path
import os,sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["LANGSMITH_TRACING"]=os.environ["LANGCHAIN_TRACING_V2"]="false"
if __name__=="__main__":
    from copper_mvp.common import PROJECT_ROOT
    from copper_mvp.data import DataRepository
    from copper_mvp.tabular_controls import run_matched_controls
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id",required=True)
    parser.add_argument("--request-key",required=True)
    parser.add_argument("--run-root",type=Path,default=Path("runs/mvp"))
    args=parser.parse_args()
    if not args.run_root.resolve().is_relative_to((PROJECT_ROOT/"runs").resolve()):
        parser.error("Private artifacts must remain under local runs/")
    root,result=run_matched_controls(DataRepository(),args.run_root,args.study_id,args.request_key,
        progress=lambda value:print(json.dumps(value),flush=True))
    print(json.dumps({"status":result["status"],"output":str(root)}),flush=True)
