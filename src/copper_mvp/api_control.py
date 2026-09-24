"""Workbench APIs for reviewing exact Mock proposals and execution evidence."""

from fastapi import APIRouter, Request

from copper_mvp.common import WorkbenchError
from copper_mvp.control.contracts import DEVICE, NOTICE, POLICY, ApprovalInput, FaultInput, ProposalInput, TickInput


def control_router():
    router = APIRouter(prefix="/api/v2")

    def service(request):
        return request.app.state.workbench.control

    def actor(request, permission="read"):
        value = getattr(request.state, "principal", None)
        if value is None:
            raise WorkbenchError("需要本机访问码", "UNAUTHENTICATED")
        value.require(permission)
        return value

    @router.get("/control")
    def info(request: Request):
        principal = actor(request)
        client = service(request).client
        return {
            "environment": "MOCK",
            "notice": NOTICE,
            "device_id": DEVICE,
            "policy_ref": POLICY,
            "connected": client is not None,
            "can_control": principal.role == "owner",
            "can_test_admin": principal.role == "owner" and bool(client and client.admin_key),
            "operations": ["set_absolute"],
        }

    @router.get("/mock/devices/{device_id}/state")
    def state(request: Request, device_id: str):
        actor(request)
        if device_id != DEVICE:
            raise WorkbenchError("合成设备不存在", "DEVICE_NOT_FOUND")
        return service(request).require_client().read_state()

    @router.get("/commands")
    def commands(request: Request):
        return {"items": service(request).list(actor(request))}

    @router.post("/commands", status_code=201)
    def propose(request: Request, payload: ProposalInput):
        return service(request).propose(actor(request, "mock_control"), payload)

    @router.get("/commands/{identifier}")
    def command(request: Request, identifier: str):
        return service(request).get(actor(request), identifier)

    @router.post("/commands/{identifier}/approvals")
    def approve(request: Request, identifier: str, payload: ApprovalInput):
        return service(request).approve(actor(request, "approve"), identifier, payload)

    @router.post("/commands/{identifier}/cancel")
    def cancel(request: Request, identifier: str):
        return service(request).cancel(actor(request, "mock_control"), identifier)

    @router.post("/commands/{identifier}/revoke")
    def revoke(request: Request, identifier: str):
        return service(request).cancel(actor(request, "approve"), identifier, revoke=True)

    @router.post("/commands/{identifier}/reconcile")
    def reconcile(request: Request, identifier: str):
        principal = actor(request, "mock_control")
        service(request).get(principal, identifier)
        service(request).step()
        return service(request).get(principal, identifier)

    @router.post("/mock/test-admin/tick")
    def tick(request: Request, payload: TickInput):
        actor(request, "manage")
        result = service(request).require_client().tick(payload.seconds)
        service(request).step()
        return result

    @router.post("/mock/test-admin/faults")
    def fault(request: Request, payload: FaultInput):
        actor(request, "manage")
        return service(request).require_client().inject_fault(payload.model_dump(exclude_none=True))

    return router
