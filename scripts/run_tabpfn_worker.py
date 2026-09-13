"""Offline request worker with a single cached TabPFN context."""
from pathlib import Path
import sys,json,contextlib,os,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from copper_mvp.tabpfn_runtime import prepare_tabpfn_runtime,disable_worker_network
from copper_mvp.common import PROJECT_ROOT,file_hash
from copper_mvp.tabpfn_registry import WEIGHT_SHA256
with contextlib.redirect_stdout(sys.stderr):
    runtime=prepare_tabpfn_runtime()
    disable_worker_network()
    import numpy as np
    import torch
    from tabpfn import TabPFNRegressor
    from copper_mvp.tabpfn_attention import enable_chunked_cpu_attention
    enable_chunked_cpu_attention()
torch.set_num_threads(1)
cache_id=None
models=[]
weight=Path(os.environ.get("COPPER_TABPFN_WEIGHT",PROJECT_ROOT/"runs/dependencies/tabpfn-weights/tabpfn-v2-regressor-v2_default.ckpt"))
assert weight.is_file() and file_hash(weight)==WEIGHT_SHA256


def local_path(raw):
    path=Path(raw).resolve()
    if not path.is_relative_to((PROJECT_ROOT/"runs").resolve()):
        raise ValueError("Worker files must remain under project runs")
    return path


for line in sys.stdin:
    request=json.loads(line)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            action=request["action"]
            if action=="fit":
                if request["context_id"]==cache_id:
                    result={"cache_hit":True,"runtime":runtime}
                else:
                    path=local_path(request["input_path"])
                    assert file_hash(path)==request["input_sha256"]
                    data=np.load(path,allow_pickle=False)
                    X,y=data["X"],data["y"]
                    actual_context=hashlib.sha256(X.tobytes()+y.tobytes()+str(request["seed"]).encode()+WEIGHT_SHA256.encode()+str(request["n_estimators"]).encode()).hexdigest()
                    assert actual_context==request["context_id"]
                    assert len(X)<=10000 and X.shape[1]==8 and y.shape==(len(X),2)
                    start=time.perf_counter();models=[]
                    for target in (0,1):
                        model=TabPFNRegressor(n_estimators=request["n_estimators"],model_path=weight,
                            device="cpu",n_jobs=1,random_state=request["seed"]+target,
                            inference_precision=torch.float32,fit_mode="fit_with_cache",
                            memory_saving_mode=True,ignore_pretraining_limits=True)
                        model.fit(X,y[:,target]);models.append(model)
                    cache_id=request["context_id"]
                    result={"cache_hit":False,"runtime":runtime,"fit_seconds":time.perf_counter()-start,
                            "training_rows":len(X),"weight_sha256":WEIGHT_SHA256}
            elif action=="predict":
                if request["context_id"]!=cache_id:raise ValueError("Requested context is not loaded")
                path=local_path(request["input_path"]);assert file_hash(path)==request["input_sha256"]
                X=np.load(path,allow_pickle=False)["X"]
                start=time.perf_counter()
                outputs=[m.predict(X,output_type="main",quantiles=[.1,.5,.9]) for m in models]
                mean=np.column_stack([o["mean"] for o in outputs])
                quantiles=np.stack([np.asarray(o["quantiles"]).T for o in outputs],axis=-1)
                assert np.isfinite(mean).all() and np.isfinite(quantiles).all()
                target=local_path(request["output_path"])
                np.savez(target,mean=mean,quantiles=quantiles)
                result={"sha256":file_hash(target),"prediction_seconds":time.perf_counter()-start}
            else:raise ValueError("Unknown worker request")
        response={"request_id":request["request_id"],"status":"ok","result":result}
    except Exception as exc:
        response={"request_id":request.get("request_id"),"status":"error","error":type(exc).__name__+": "+str(exc)[:300]}
    print("TABPFN_JSON "+json.dumps(response),flush=True)
