"""Authenticated optimizer portfolio comparisons and per-run evidence."""
import json
import threading
from fastapi import APIRouter,Request
from fastapi.responses import FileResponse
from copper_mvp.common import WorkbenchError
from copper_mvp.portfolio_studies import PortfolioStudies
from copper_mvp.portfolio_comparison import PortfolioRequest,verify_run


def portfolio_router():
    router=APIRouter(prefix="/api/v2/optimizer-portfolios");lock=threading.Lock()
    def service(request):
        wb=request.app.state.workbench
        with lock:
            if not hasattr(wb,"_portfolio_studies"):
                wb._portfolio_studies=PortfolioStudies(wb.root,wb.data,wb.models)
        return wb._portfolio_studies
    def actor(request,permission="read"):
        value=request.state.principal;value.require(permission);return value

    @router.post("",status_code=202)
    def create(request:Request,payload:PortfolioRequest):
        return service(request).submit(actor(request,"compute"),payload,request.app.state.workbench.executor)

    @router.get("")
    def history(request:Request):
        return {"items":service(request).list(actor(request))}

    @router.get("/{identifier}")
    def study(request:Request,identifier:str):
        return service(request).get(actor(request),identifier)

    @router.post("/{identifier}/resume",status_code=202)
    def resume(request:Request,identifier:str):
        return service(request).resume(actor(request,"compute"),identifier,request.app.state.workbench.executor)

    @router.get("/{identifier}/runs")
    def run(request:Request,identifier:str,case:int,seed:int,strategy:str):
        current=service(request);state=current.get(actor(request),identifier)
        rows=(state.get("result") or {}).get("results",[])
        selected=next((r for r in rows if r["case"]==case and r["seed"]==seed and r["strategy"]==strategy),None)
        if selected is None:raise WorkbenchError("运行记录不存在","RUN_NOT_FOUND")
        root=current.directory(identifier)
        path=(root/selected["result_path"]).resolve()
        if not path.is_relative_to(root.resolve()):raise WorkbenchError("工件路径无效","SOURCE_CHANGED")
        result=json.loads(path.read_text(encoding="utf-8"))
        if result["status"]!="failed":
            verify_run(path.parent,result)
            result["allocation_trace"]=json.loads((path.parent/"allocation_trace.json").read_text(encoding="utf-8"))
        return result

    @router.get("/{identifier}/export")
    def export(request:Request,identifier:str):
        current=service(request);state=current.get(actor(request),identifier)
        if state["status"] not in ("completed","completed_with_failures"):raise WorkbenchError("研究尚未完成","TASK_STATE")
        return FileResponse(current.directory(identifier)/"report.md",filename="portfolio-"+identifier+".md")
    return router
