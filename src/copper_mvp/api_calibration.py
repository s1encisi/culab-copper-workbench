"""Authenticated calibration study and interval replay API."""
from fastapi import APIRouter,Request
from pydantic import BaseModel,ConfigDict,Field

from copper_mvp.calibration_contracts import CalibrationRequest
from copper_mvp.common import WorkbenchError


class CalibrationPrediction(BaseModel):
    model_config=ConfigDict(extra="forbid")
    event_id:str=Field(min_length=1,max_length=180)
    method_id:str=Field(min_length=1,max_length=80)
    seed:int|None=None


def calibration_router():
    router=APIRouter(prefix="/api/v2/calibrations")

    def context(request):
        actor=getattr(request.state,"principal",None)
        if actor is None:raise WorkbenchError("需要本机访问码","UNAUTHENTICATED")
        wb=request.app.state.workbench
        return wb,actor

    @router.get("")
    def list_studies(request:Request):
        wb,actor=context(request)
        return {"items":wb.calibrations.list(actor)}

    @router.post("",status_code=202)
    def create(request:Request,payload:CalibrationRequest):
        wb,actor=context(request)
        return wb.calibrations.submit(actor,payload,wb.executor)

    @router.get("/{identifier}")
    def study(request:Request,identifier:str):
        wb,actor=context(request)
        return wb.calibrations.get(actor,identifier)

    @router.post("/{identifier}/predict")
    def predict(request:Request,identifier:str,payload:CalibrationPrediction):
        wb,actor=context(request)
        return wb.calibrations.predict(actor,identifier,**payload.model_dump())

    return router
