"""Compare matched local prediction paths after a model has been trained."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
import uuid

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from copper_mvp.checks import verify_prediction
from copper_mvp.common import DEFAULT_RUNS_DIR, PROJECT_ROOT, digest, utc_now, write_json
from copper_mvp.contracts import RunRequest
from copper_mvp.data import DataRepository
from copper_mvp.modeling import ModelManager
from copper_mvp.workflows import Workbench


def main():
    data = DataRepository()
    output = PROJECT_ROOT / "runs/benchmarks" / uuid.uuid4().hex
    workbench = Workbench(output, data)
    workbench.models = ModelManager(DEFAULT_RUNS_DIR, data)
    bundle = workbench.models.manifest()["bundle_id"]
    events = data.events(scope="dual", limit=8)["items"]

    def fixed(event, key):
        request = RunRequest(task_type="predict", event_id=event, request_key=key, bundle_id=bundle).model_dump()
        run, _ = workbench.store.create(request, digest(request))
        rid = run["run_id"]; workbench.store.start(rid)
        started = perf_counter()
        def step(name, action):
            t = perf_counter(); stamp = utc_now(); value = action()
            workbench.store.trace(rid, name, "completed", stamp, (perf_counter()-t)*1000, {"result":"完成"})
            return value
        step("A1 数据装配", lambda:data.context(event))
        step("A2 工况", lambda:data.mode_cards[event])
        result = step("A4 数值预测", lambda:workbench.models.predict(event, bundle_id=bundle))
        draft = workbench.store.directory(rid) / "draft_result.json"
        write_json(draft, result)
        result["audit"] = step("A5 结果复核", lambda:verify_prediction(result, data.context(event)))
        write_json(draft, result)
        workbench.store.finish(rid, result, (perf_counter()-started)*1000)
        return result

    def graph(event, key):
        created = workbench.submit(RunRequest(task_type="predict", event_id=event, request_key=key, bundle_id=bundle))
        workbench.futures[created["run_id"]].result(timeout=30)
        run = workbench.store.get(created["run_id"])
        if run["status"] != "completed": raise RuntimeError(run["error"])
        return run["result"]

    measurements=[]; differences=[]
    try:
        workbench.executor.submit(fixed, events[0]["event_id"], "warm-fixed").result()
        graph(events[0]["event_id"], "warm-graph")
        for repeat in range(3):
            for i, event in enumerate(events):
                results={}
                for arm in (["fixed","graph"] if (repeat+i)%2==0 else ["graph","fixed"]):
                    key=f"{arm}-{repeat}-{i}"; start=perf_counter()
                    results[arm] = workbench.executor.submit(fixed,event["event_id"],key).result() if arm=="fixed" else graph(event["event_id"],key)
                    measurements.append({"arm":arm,"repeat":repeat,"event_id":event["event_id"],"elapsed_ms":(perf_counter()-start)*1000})
                for target in ("cu","as"):
                    differences.append(abs(results['fixed']['predictions'][target]['value']-results['graph']['predictions'][target]['value']))
    finally:
        workbench.close()
    summaries=[]
    for arm in ("fixed","graph"):
        values=np.array([r['elapsed_ms'] for r in measurements if r['arm']==arm])
        summaries.append({"arm":arm,"runs":len(values),"p50_ms":float(np.median(values)),"p95_ms":float(np.quantile(values,.95)),"throughput_runs_per_second":float(len(values)/values.sum()*1000),"concurrency":1,"failures":0,"llm_calls":0,"tokens":0,"api_cost_cny":0})
    report={"created_at":utc_now(),"model_bundle":bundle,"sample":"最近8个双段OOF事件，3轮交替顺序，双臂各1次预热","common_work":"相同事件、模型、检查、运行索引、节点记录和结果文件；图路径另存 LangGraph checkpoint","max_prediction_difference":max(differences),"summary":summaries,"measurements":measurements}
    write_json(output/'benchmark.json',report)
    print({"report":str(output/'benchmark.json'),"max_prediction_difference":max(differences),"summary":summaries})


if __name__ == '__main__':
    main()
