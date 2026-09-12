"""Private portfolio study jobs with resumable completed-run artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading

from copper_mvp.common import WorkbenchError,digest,file_hash,utc_now,write_json
from copper_mvp.model_comparisons import process_alive
from copper_mvp.portfolio_comparison import PortfolioRequest,PORTFOLIO_FILES,source_record,compare_portfolios
from copper_mvp.optimizer_portfolio import POLICY


class PortfolioStudies:
    def __init__(self,root,data,models):
        self.root=Path(root)/"optimizer_portfolios";self.root.mkdir(parents=True,exist_ok=True)
        self.data,self.models=data,models
        self.lock=threading.RLock()

    def directory(self,identifier):
        if len(identifier)!=32 or any(c not in "0123456789abcdef" for c in identifier):
            raise WorkbenchError("组合研究不存在","PORTFOLIO_NOT_FOUND")
        return self.root/identifier

    def fingerprint(self,request):
        source,_=source_record(self.data,self.models,request)
        return digest({"request":request.model_dump(),"source":source,"policy":POLICY,
                       "code":{n:file_hash(Path(__file__).with_name(n)) for n in PORTFOLIO_FILES}})

    def submit(self,actor,request,executor=None):
        actor.require("compute")
        identifier=digest([actor.project_id,actor.user_id,request.request_key])[:32]
        root=self.directory(identifier);fingerprint=self.fingerprint(request)
        with self.lock:
            if root.exists():
                state=self.get(actor,identifier,result=False)
                if state["fingerprint"]!=fingerprint:raise WorkbenchError("请求键已用于不同配置或来源","REQUEST_CONFLICT")
                return {**state,"reused":True}
            root.mkdir()
            state={"id":identifier,"project_id":actor.project_id,"owner_id":actor.user_id,"owner_pid":os.getpid(),
                "request":request.model_dump(mode="json"),"fingerprint":fingerprint,"status":"queued","created_at":utc_now()}
            write_json(root/"state.json",state)
        if executor is None:
            self.execute(identifier,request);return self.get(actor,identifier)
        executor.submit(self.execute,identifier,request)
        return state

    def execute(self,identifier,request):
        root=self.directory(identifier)
        def update(**values):
            with self.lock:
                state=json.loads((root/"state.json").read_text(encoding="utf-8"));state.update(values)
                write_json(root/"state.json",state)
        update(status="running",started_at=utc_now())
        try:
            result=compare_portfolios(self.data,self.models,request,root,lambda value:update(progress=value))
            update(status=result["status"],finished_at=utc_now(),comparison_sha256=file_hash(root/"comparison.json"),runs=result["runs"])
        except Exception as exc:
            update(status="failed",finished_at=utc_now(),error={"code":getattr(exc,"code",type(exc).__name__),"message":str(exc)[:300]})

    def get(self,actor,identifier,result=True):
        actor.require("read");root=self.directory(identifier)
        if not (root/"state.json").exists():raise WorkbenchError("组合研究不存在","PORTFOLIO_NOT_FOUND")
        state=json.loads((root/"state.json").read_text(encoding="utf-8"))
        if state["project_id"]!=actor.project_id or (state["owner_id"]!=actor.user_id and actor.role!="owner"):
            raise WorkbenchError("无权访问该组合研究","FORBIDDEN")
        if state["status"] in ("queued","running") and not process_alive(state["owner_pid"]):
            state={**state,"status":"interrupted"}
        if result and state["status"] in ("completed","completed_with_failures"):
            if file_hash(root/"comparison.json")!=state["comparison_sha256"]:raise WorkbenchError("比较结果哈希不符","SOURCE_CHANGED")
            value=json.loads((root/"comparison.json").read_text(encoding="utf-8"))
            if file_hash(root/"protocol.json")!=value["protocol_sha256"]:raise WorkbenchError("比较协议哈希不符","SOURCE_CHANGED")
            for name,expected in value["run_hashes"].items():
                path=(root/name).resolve()
                if not path.is_relative_to(root.resolve()) or file_hash(path)!=expected:
                    raise WorkbenchError("运行结果哈希不符","SOURCE_CHANGED")
            state["result"]=value
        return state

    def list(self,actor):
        actor.require("read");items=[]
        for path in self.root.glob("*/state.json"):
            state=json.loads(path.read_text(encoding="utf-8"))
            if state["project_id"]==actor.project_id and (state["owner_id"]==actor.user_id or actor.role=="owner"):
                items.append(self.get(actor,path.parent.name,result=False))
        return sorted(items,key=lambda r:r["created_at"],reverse=True)

    def resume(self,actor,identifier,executor=None):
        actor.require("compute")
        with self.lock:
            state=self.get(actor,identifier,result=False)
            if state["status"] not in ("failed","interrupted","completed_with_failures"):
                raise WorkbenchError("仅恢复中断或未全部完成的研究","TASK_STATE")
            if state["status"]=="completed_with_failures":
                completed=self.get(actor,identifier)
                if all(r["status"]!="failed" for r in completed["result"]["results"]):return {**completed,"reused":True}
            request=PortfolioRequest.model_validate(state["request"])
            if state["fingerprint"]!=self.fingerprint(request):raise WorkbenchError("代码、数据或参数已变化","SOURCE_CHANGED")
            state.update(status="queued",owner_pid=os.getpid());state.pop("error",None)
            write_json(self.directory(identifier)/"state.json",state)
        if executor is None:
            self.execute(identifier,request);return self.get(actor,identifier)
        executor.submit(self.execute,identifier,request)
        return state
