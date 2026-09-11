"""Private research jobs for mature-label routing studies."""
from __future__ import annotations
import json
import os
from pathlib import Path
import threading

from pydantic import BaseModel, ConfigDict, Field
from copper_mvp.common import WorkbenchError, digest, file_hash, utc_now, write_json
from copper_mvp.model_comparisons import ComparisonService, process_alive
from copper_mvp.routing_metrics import RoutingPolicy
from copper_mvp.routing_replay import run_replay


class RoutingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_key: str = Field(min_length=1,max_length=100,pattern=r"^[A-Za-z0-9_.:-]+$")
    comparison_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    max_wall_seconds: int = Field(default=600,ge=60,le=1800)


class RoutingStudies:
    def __init__(self, root, data):
        self.root, self.data = Path(root), data
        self.directory_root = self.root/"prediction_studies"
        self.directory_root.mkdir(parents=True,exist_ok=True)
        self.lock = threading.RLock()
        self.comparisons = ComparisonService(self.root/"model_comparisons",data)

    def directory(self, identifier):
        if len(identifier)!=32 or any(c not in "0123456789abcdef" for c in identifier):
            raise WorkbenchError("回放编号无效","ROUTING_NOT_FOUND")
        return self.directory_root/identifier

    def submit(self, actor, request, executor=None):
        actor.require("compute")
        source = self.comparisons.get(request.comparison_id, result=False)
        if source["status"] != "completed":
            raise WorkbenchError("来源模型比较尚未完成","ROUTING_SOURCE")
        identifier=digest([actor.project_id,actor.user_id,request.request_key])[:32]
        root=self.directory(identifier)
        fingerprint=digest({"request":request.model_dump(),"source":source["fingerprint"],
            "policy":RoutingPolicy().model_dump(),"code":{n:file_hash(Path(__file__).with_name(n))
            for n in ("routing_metrics.py","prediction_router.py","routing_replay.py")}})
        with self.lock:
            if root.exists():
                old=self.get(actor,identifier,result=False)
                if old["fingerprint"]!=fingerprint:
                    raise WorkbenchError("请求键已用于不同来源、策略或实现","REQUEST_CONFLICT")
                return {**old,"reused":True}
            root.mkdir()
            state={"id":identifier,"owner_id":actor.user_id,"project_id":actor.project_id,"request":request.model_dump(),
                "fingerprint":fingerprint,"status":"queued","created_at":utc_now(),"owner_pid":os.getpid(),
                "evaluation_mode":"historical_replay","live_model_change":False}
            write_json(root/"state.json",state)
        if executor is None:
            self.execute(identifier,request)
            return self.get(actor,identifier)
        executor.submit(self.execute,identifier,request)
        return {**state,"reused":False}

    def execute(self, identifier, request):
        root=self.directory(identifier)
        def update(**values):
            with self.lock:
                state=json.loads((root/"state.json").read_text(encoding="utf-8"))
                state.update(values)
                write_json(root/"state.json",state)
        update(status="running",started_at=utc_now())
        try:
            result=run_replay(self.data,self.comparisons.directory(request.comparison_id),root,
                progress=lambda value:update(progress=value),max_wall_seconds=request.max_wall_seconds)
            update(status="completed",finished_at=utc_now(),evaluation_sha256=file_hash(root/"evaluation.json"),
                   common_events=result["common_events"])
        except Exception as exc:
            update(status="failed",finished_at=utc_now(),error={"code":getattr(exc,"code",type(exc).__name__),
                "message":str(exc)[:300]})

    def get(self, actor, identifier, *, result=True):
        actor.require("read")
        root=self.directory(identifier)
        if not (root/"state.json").exists():
            raise WorkbenchError("回放记录不存在","ROUTING_NOT_FOUND")
        state=json.loads((root/"state.json").read_text(encoding="utf-8"))
        if state["project_id"]!=actor.project_id or (state["owner_id"]!=actor.user_id and actor.role!="owner"):
            raise WorkbenchError("无权访问此回放","FORBIDDEN")
        if state["status"] in ("queued","running") and not process_alive(state["owner_pid"]):
            state={**state,"status":"interrupted"}
        if result and state["status"]=="completed":
            if file_hash(root/"evaluation.json")!=state["evaluation_sha256"]:
                raise WorkbenchError("回放结果哈希不符","ROUTING_HASH_MISMATCH")
            state["result"]=json.loads((root/"evaluation.json").read_text(encoding="utf-8"))
            if file_hash(root/"replay_manifest.json")!=state["result"]["replay_manifest_sha256"]:
                raise WorkbenchError("回放账本清单已改变","ROUTING_HASH_MISMATCH")
            manifest=json.loads((root/"replay_manifest.json").read_text(encoding="utf-8"))
            for name,key in (("protocol.json","protocol_sha256"),("predictions.csv","predictions_sha256"),("decisions.jsonl","decisions_sha256")):
                if file_hash(root/name)!=manifest[key]:
                    raise WorkbenchError("回放工件与清单不一致", "ROUTING_HASH_MISMATCH")
        return state

    def list(self, actor):
        actor.require("read")
        result=[]
        for path in self.directory_root.glob("*/state.json"):
            state=json.loads(path.read_text(encoding="utf-8"))
            if state["project_id"]==actor.project_id and (state["owner_id"]==actor.user_id or actor.role=="owner"):
                result.append(self.get(actor,path.parent.name,result=False))
        return sorted(result,key=lambda r:r["created_at"],reverse=True)

    def decisions(self, actor, identifier, event_id):
        state=self.get(actor,identifier)
        if state["status"]!="completed":
            raise WorkbenchError("回放尚未完成", "ROUTING_NOT_READY")
        root=self.directory(identifier)
        manifest=json.loads((root/"replay_manifest.json").read_text(encoding="utf-8"))
        if file_hash(root/"decisions.jsonl")!=manifest["decisions_sha256"]:
            raise WorkbenchError("决策账本哈希不符","ROUTING_HASH_MISMATCH")
        rows=[]
        with (root/"decisions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                value=json.loads(line)
                if value["event_id"]==event_id:
                    rows.append(value)
        if not rows:
            raise WorkbenchError("该回放没有此事件","EVENT_NOT_FOUND")
        evidence={}
        for value in rows:
            for reference in [value["metric_snapshot_id"],*value.get("confirmation_snapshots",[])]:
                snapshot=json.loads((root/"metric_snapshots"/(reference+".json")).read_text(encoding="utf-8"))
                identity=snapshot.pop("id")
                if digest(snapshot)!=identity or identity!=reference:
                    raise WorkbenchError("指标快照哈希不符","ROUTING_HASH_MISMATCH")
                evidence[identity]={"id":identity,**snapshot}
        return {"items":rows,"snapshots":evidence,"live_model_change":False,
                "batch_recorded_at":manifest.get("created_at"),"evaluation_mode":"historical_replay"}
