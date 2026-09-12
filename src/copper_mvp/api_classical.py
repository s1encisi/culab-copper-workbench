"""Read the grouped G6 classical inventory runs without exposing local paths."""
import json
from pathlib import Path
from fastapi import APIRouter,Request
from fastapi.responses import FileResponse
from copper_mvp.common import WorkbenchError,file_hash
from copper_mvp.model_comparisons import process_alive


def classical_router():
    router=APIRouter(prefix="/api/v2/classical-studies")

    def folder(request,identifier):
        request.state.principal.require("read")
        if len(identifier)!=32 or any(c not in "0123456789abcdef" for c in identifier):
            raise WorkbenchError("模型目录研究不存在","CLASSICAL_STUDY_NOT_FOUND")
        root=request.app.state.workbench.root/"classical_studies"/identifier
        if not (root/"state.json").exists():raise WorkbenchError("模型目录研究不存在","CLASSICAL_STUDY_NOT_FOUND")
        return root

    def state(root,result=False):
        value=json.loads((root/"state.json").read_text(encoding="utf-8"))
        if value["status"]=="running" and not process_alive(value.get("owner_pid")):
            value={**value,"status":"interrupted"}
        if result and value["status"] in ("completed","completed_with_missing_predictions"):
            if file_hash(root/"evaluation.json")!=value["evaluation_sha256"]:
                raise WorkbenchError("研究评价文件哈希不符","SOURCE_CHANGED")
            evaluation=json.loads((root/"evaluation.json").read_text(encoding="utf-8"))
            if file_hash(root/"protocol.json")!=evaluation["protocol_sha256"] or file_hash(root/"predictions.csv")!=evaluation["predictions_sha256"]:
                raise WorkbenchError("研究协议或预测账本哈希不符","SOURCE_CHANGED")
            value["result"]=evaluation
        return value

    @router.get("")
    def studies(request:Request):
        request.state.principal.require("read")
        root=request.app.state.workbench.root/"classical_studies"
        return {"items":sorted([state(p.parent) for p in root.glob("*/state.json")],key=lambda v:v["created_at"],reverse=True)}

    @router.get("/{identifier}")
    def study(request:Request,identifier:str):
        return state(folder(request,identifier),True)

    @router.get("/{identifier}/export")
    def export(request:Request,identifier:str):
        root=folder(request,identifier);current=state(root,True)
        if current["status"] not in ("completed","completed_with_missing_predictions"):
            raise WorkbenchError("研究尚未完成","TASK_STATE")
        return FileResponse(root/"report.md",filename="classical-study-"+identifier+".md")
    return router
