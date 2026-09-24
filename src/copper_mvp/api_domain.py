from fastapi import APIRouter, Request

from copper_mvp.domain_evaluation import EvaluationRequest, EvaluationReview


def domain_router():
    router = APIRouter(prefix="/api/v2/domain-evaluations")

    @router.get("/defaults")
    def defaults(request: Request):
        request.state.principal.require("read")
        return request.app.state.workbench.domain_adaptation.defaults()

    @router.get("")
    def history(request: Request):
        return {"items": request.app.state.workbench.domain_adaptation.list(request.state.principal)}

    @router.post("", status_code=201)
    def create(request: Request, payload: EvaluationRequest):
        return request.app.state.workbench.domain_adaptation.create(request.state.principal, payload)

    @router.get("/{identifier}")
    def get(request: Request, identifier: str):
        return request.app.state.workbench.domain_adaptation.get(request.state.principal, identifier)

    @router.post("/{identifier}/run", status_code=202)
    def run(request: Request, identifier: str):
        return request.app.state.workbench.domain_adaptation.run(request.state.principal, identifier)

    @router.post("/{identifier}/review")
    def review(request: Request, identifier: str, payload: EvaluationReview):
        return request.app.state.workbench.domain_adaptation.review(request.state.principal, identifier, payload)

    return router
