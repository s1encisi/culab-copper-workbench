"""Read-only workspace summaries backed by existing task and budget records."""
import json

from fastapi import APIRouter,Request
from copper_mvp.access import PROJECT
from copper_mvp.common import APP_VERSION,WorkbenchError,utc_now


def workspace_router():
    router=APIRouter(prefix="/api/v2/workspace")
    def context(request):
        actor=getattr(request.state,"principal",None)
        if actor is None:raise WorkbenchError("需要本机访问码","UNAUTHENTICATED")
        actor.require("read")
        if actor.project_id!=PROJECT:raise WorkbenchError("项目不可访问","FORBIDDEN")
        return request.app.state.workbench,actor

    @router.get("")
    def summary(request:Request):
        wb,actor=context(request)
        with wb.store.connection() as c:
            own=c.execute("SELECT status,COUNT(*) n FROM research_tasks WHERE owner_id=? GROUP BY status",(actor.user_id,)).fetchall()
            if actor.role=="owner":
                budget=c.execute("SELECT COUNT(*) calls,COALESCE(SUM(spent),0) spent,COALESCE(SUM(CASE WHEN spent IS NULL THEN reserved ELSE 0 END),0) reserved FROM budget").fetchone()
                scope="project"
            else:
                budget=c.execute("""SELECT COUNT(*) calls,COALESCE(SUM(b.spent),0) spent,
                    COALESCE(SUM(CASE WHEN b.spent IS NULL THEN b.reserved ELSE 0 END),0) reserved
                    FROM budget b JOIN research_calls r ON r.call_id=b.call_id WHERE r.owner_id=?""",(actor.user_id,)).fetchone()
                scope="own_research"
        return {"version":APP_VERSION,"project_id":PROJECT,"environment":"historical_research","observed_at":utc_now(),
                "user":{"id":actor.user_id,"role":actor.role},
                "assistant_live":wb.research.allow_live,"mock_configured":wb.control.client is not None,
                "research_tasks":{row["status"]:row["n"] for row in own},
                "budget":{**dict(budget),"currency":"CNY","scope":scope},
                "capabilities":{"knowledge":True,"calibration":True,"model_lifecycle":True,"domain_adaptation":hasattr(wb,"domain_adaptation")}}

    @router.get("/tasks")
    def tasks(request:Request):
        wb,actor=context(request)
        with wb.store.connection() as c:
            research=c.execute("""SELECT id,status,version,created_at,elapsed_ms,model_calls,tool_calls,request_json,error_json
                                  FROM research_tasks WHERE owner_id=? ORDER BY created_at DESC LIMIT 100""",(actor.user_id,)).fetchall()
        values=[]
        for row in research:
            body=json.loads(row["request_json"])
            values.append({"id":row["id"],"kind":"research","title":body["question"],"status":row["status"],
                           "created_at":row["created_at"],"elapsed_ms":row["elapsed_ms"],"version":row["version"],
                           "model_calls":row["model_calls"],"tool_calls":row["tool_calls"],
                           "error":json.loads(row["error_json"]) if row["error_json"] else None})
        run_labels={"train":"模型训练","predict":"历史事件预测","optimize":"多目标优化","diagnostic":"异常路径检查","agent_diagnostic":"优化诊断"}
        for row in wb.store.list(limit=100):
            values.append({"id":row["run_id"],"kind":"run","title":run_labels.get(row["task_type"],row["task_type"]),
                           "task_type":row["task_type"],"status":row["status"],"created_at":row["created_at"],
                           "elapsed_ms":row.get("duration_ms"),"error":row.get("error")})
        for folder,kind in (("model_comparisons","model_comparison"),("optimizer_comparisons","optimizer_comparison"),("calibration_studies","calibration")):
            for path in sorted((wb.root/folder).glob("*/state.json"),key=lambda p:p.stat().st_mtime,reverse=True)[:60]:
                state=json.loads(path.read_text(encoding="utf-8"))
                values.append({"id":state.get("run_id",state.get("id",path.parent.name)),"kind":kind,
                               "title":state.get("request",{}).get("request_key",path.parent.name),
                               "status":state["status"],"created_at":state.get("created_at",""),
                               "progress":state.get("progress"),"error":state.get("error")})
        return {"items":sorted(values,key=lambda row:row.get("created_at") or "",reverse=True)[:180]}

    @router.get("/research-tasks/{task_id}/events")
    def task_events(request:Request,task_id:str):
        wb,actor=context(request)
        return {"items":wb.research.store.events(actor,task_id)}

    @router.get("/governance")
    def governance(request:Request):
        wb,actor=context(request)
        with wb.store.connection() as c:
            rows=c.execute("""SELECT e.kind,e.payload_json,e.task_id FROM research_events e
                              JOIN research_tasks t ON t.id=e.task_id WHERE t.owner_id=? ORDER BY e.seq DESC LIMIT 100""",(actor.user_id,)).fetchall()
            usage=c.execute("""SELECT r.usage_json FROM research_calls r WHERE r.owner_id=? AND r.outcome='known'""",(actor.user_id,)).fetchall()
        usage=[json.loads(v[0]) for v in usage if v[0]]
        return {"scope":"own_research","events":[{"kind":v["kind"],"task_id":v["task_id"],"detail":json.loads(v["payload_json"])} for v in rows],
                "usage":{"calls":len(usage),"input_tokens":sum(v.get("input_tokens",0) for v in usage),
                         "output_tokens":sum(v.get("output_tokens",0) for v in usage),
                         "estimated_cost_cny":sum(v.get("estimated_cost_cny",0) for v in usage)},
                "live_calls_enabled":wb.research.allow_live}

    return router
