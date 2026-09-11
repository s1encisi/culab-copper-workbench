"""Synthetic protocol acceptance: independent outcomes, faults and crash recovery."""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient
import pytest

from copper_mvp.access import AccessControl, Principal
from copper_mvp.common import WorkbenchError
from copper_mvp.control.client import MockClient
from copper_mvp.control.commands import CommandService
from copper_mvp.control.contracts import DEVICE, POINTS
from copper_mvp.control.mock import MockDevice, canonical_hash
from copper_mvp.control.mock_api import create_mock_app
from copper_mvp.research_store import ResearchStore
from copper_mvp.storage import RunStore


class DeviceLink:
    admin_key = True
    def __init__(self, device):
        self.device = device
        self.crash_after_send = False
        self.submissions = 0

    def __getattr__(self, name):
        return getattr(self.device, name)

    def close(self):
        pass

    def submit_command(self, *args):
        self.submissions += 1
        receipt = self.device.submit_command(*args)
        if self.crash_after_send:
            raise SystemExit("synthetic worker crash after durable device acceptance")
        if receipt["ack_lost"]:
            raise WorkbenchError("合成 ACK 丢失", "MOCK_TRANSPORT")
        return {**receipt, "ack_received": receipt["ack_at"] <= self.device.read_state()["virtual_time"]}


@pytest.fixture
def system(tmp_path):
    store = RunStore(tmp_path / "workbench")
    access = AccessControl(ResearchStore(store))
    actor = access.authenticate(key=access.owner_key_path.read_text().strip())
    device = MockDevice(tmp_path / "independent_mock")
    link = DeviceLink(device)
    service = CommandService(store, access, link, autostart=False)
    yield service, actor, device, link
    service.close()


def propose(service, actor, *, key="synthetic-1", targets=None, **kwargs):
    return service.propose(actor, {"request_key": key, "reason": "合成 100 A 到 105 A 协议验收",
                        "targets": targets or [{"point_id": POINTS[0], "value": 105, "unit": "A"}], **kwargs})


def approve(service, actor, proposal):
    return service.approve(actor, proposal["id"], {"payload_hash": proposal["payload_hash"]})


def advance(service, actor, device, identifier, count=10):
    for _ in range(count):
        device.tick(1)
        service.step()
        result = service.get(actor, identifier)
        if result["status"] in {"VERIFIED", "FAILED", "PARTIAL", "REJECTED"}:
            return result
    return result


def test_exact_command_receipt_ramp_and_three_samples(system):
    service, actor, device, link = system
    proposal = propose(service, actor)
    assert propose(service, actor)["id"] == proposal["id"]
    with pytest.raises(WorkbenchError, match="请求键"):
        propose(service, actor, targets=[{"point_id": POINTS[0], "value": 104}])
    # Ordinary telemetry does not invalidate approval.
    version = device.read_state()["state_version"]
    device.tick(.5)
    assert device.read_state()["state_version"] == version
    approve(service, actor, proposal)
    service.step()
    accepted = service.get(actor, proposal["id"])
    assert accepted["status"] == "VERIFYING"
    assert accepted["result"]["ack_received"]
    assert device.read_state()["points"][POINTS[0]]["sp"] == 100
    for _ in range(4):
        service.step()
    assert service.get(actor, proposal["id"])["result"]["points"][POINTS[0]]["good_samples"] == 0
    outcome = advance(service, actor, device, proposal["id"])
    assert outcome["status"] == "VERIFIED"
    assert outcome["result"]["points"][POINTS[0]]["good_samples"] == 3
    record = device.get_command_status(proposal["id"])
    assert record["write_count"] == 1 and link.submissions == 1
    assert device.submit_command(proposal["payload"], proposal["payload_hash"], 1, time.time()+2)["reused"]
    altered = {**proposal["payload"], "reason": "different command"}
    with pytest.raises(WorkbenchError, match="幂等键"):
        device.submit_command(altered, canonical_hash(altered), 2, time.time()+2)


@pytest.mark.parametrize("fault,code", [
    ({"mode": "manual"}, "MOCK_INTERLOCK"), ({"mode": "maintenance"}, "MOCK_INTERLOCK"),
    ({"connected": False}, "MOCK_INTERLOCK"), ({"estop_latched": True}, "MOCK_INTERLOCK"),
    ({"interlocks_ok": False}, "MOCK_INTERLOCK"), ({"bad_quality": True}, "MOCK_QUALITY"),
    ({"stale": True}, "MOCK_QUALITY")])
def test_device_conditions_reject_without_submission(system, fault, code):
    service, actor, device, link = system
    device.inject_fault(fault)
    device.tick(2)
    with pytest.raises(WorkbenchError) as caught:
        propose(service, actor)
    assert caught.value.code == code and link.submissions == 0
    if fault.get("estop_latched"):
        with pytest.raises(WorkbenchError):
            device.inject_fault({"estop_latched": False})


def test_changed_state_payload_permission_and_ttl_invalidate_approval(system, monkeypatch):
    service, actor, device, link = system
    proposal = propose(service, actor)
    approve(service, actor, proposal)
    device.inject_fault({"mode": "manual"})
    service.step()
    assert service.get(actor, proposal["id"])["result"]["code"] == "STALE_PROPOSAL"
    device.inject_fault({"mode": "remote"})
    proposal = propose(service, actor, key="tamper")
    approve(service, actor, proposal)
    with service.connection() as c:
        payload = proposal["payload"]
        payload["request"]["targets"][0]["value"] = 103
        c.execute("UPDATE control_commands SET payload=? WHERE id=?", (json.dumps(payload), proposal["id"]))
    service.step()
    assert service.get(actor, proposal["id"])["result"]["code"] == "PAYLOAD_CONFLICT"
    key = service.access.issue(actor, "operator", "owner")
    operator = service.access.authenticate(key=key)
    proposal = propose(service, operator, key="revoked")
    approve(service, actor, proposal)
    service.access.revoke(actor, key)
    service.step()
    assert service.get(actor, proposal["id"])["result"]["code"] == "FORBIDDEN"
    proposal = propose(service, actor, key="expired", ttl_seconds=1)
    approve(service, actor, proposal)
    future = time.time() + 2
    with monkeypatch.context() as clock:
        clock.setattr(time, "time", lambda: future)
        service.step()
    assert service.get(actor, proposal["id"])["status"] == "EXPIRED"
    assert link.submissions == 0


def test_limits_roles_and_fencing(system):
    service, actor, device, link = system
    for value, ramp in [(201, 1), (111, 10), (105, .1)]:
        with pytest.raises(WorkbenchError):
            propose(service, actor, targets=[{"point_id": POINTS[0], "value": value}], ramp_seconds=ramp)
    with pytest.raises(WorkbenchError):
        propose(service, Principal("viewer", "viewer"))
    proposal = propose(service, actor)
    with pytest.raises(WorkbenchError):
        service.approve(Principal("r", "researcher"), proposal["id"], {"payload_hash": proposal["payload_hash"]})
    with pytest.raises(WorkbenchError, match="租约"):
        device.submit_command(proposal["payload"], proposal["payload_hash"], 1, time.time()-1)
    assert link.submissions == 0


def test_ack_lost_is_reconciled_without_resending(system):
    service, actor, device, link = system
    device.inject_fault({"ack_loss": True})
    proposal = propose(service, actor)
    approve(service, actor, proposal)
    service.step()
    assert service.get(actor, proposal["id"])["status"] == "UNKNOWN_OUTCOME"
    outcome = advance(service, actor, device, proposal["id"])
    assert outcome["status"] == "VERIFIED"
    assert outcome["result"]["ack_received"] is False
    assert link.submissions == 1 and outcome["outbox"]["attempts"] == 1
    assert device.get_command_status(proposal["id"])["write_count"] == 1


def test_worker_restart_fences_old_writer_and_retains_device_intent(system):
    service, actor, device, link = system
    proposal = propose(service, actor)
    approve(service, actor, proposal)
    link.crash_after_send = True
    with pytest.raises(SystemExit):
        service.step()
    assert service.get(actor, proposal["id"])["outbox"]["state"] == "dispatching"
    replacement = CommandService(service.store, service.access, link, autostart=False)
    replacement.step()
    assert service.get(actor, proposal["id"])["status"] == "DISPATCHING"  # Lease still belongs to first worker.
    with service.connection() as c:
        c.execute("UPDATE device_leases SET expires_at=0")
    replacement.step()
    service._finish_step(proposal["id"], 1, "VERIFIED", {"reason": "stale worker"}, "finished")
    assert replacement.get(actor, proposal["id"])["status"] == "VERIFYING"
    outcome = advance(replacement, actor, device, proposal["id"])
    assert outcome["status"] == "VERIFIED" and link.submissions == 1
    assert device.get_command_status(proposal["id"])["write_count"] == 1


@pytest.mark.parametrize("fault,expected", [
    ({"stuck_pv": True}, "FAILED"), ({"readback_bias": 5}, "FAILED"),
    ({"partial_write_count": 1}, "PARTIAL"), ({"reject_writes": True}, "REJECTED")])
def test_device_feedback_does_not_turn_ack_into_whole_command_success(system, fault, expected):
    service, actor, device, link = system
    device.inject_fault(fault)
    targets = [{"point_id": p, "value": 105} for p in POINTS]
    proposal = propose(service, actor, targets=targets)
    approve(service, actor, proposal)
    service.step()
    outcome = advance(service, actor, device, proposal["id"], 12)
    assert outcome["status"] == expected
    if expected == "PARTIAL":
        assert device.read_state()["points"][POINTS[1]]["sp"] == 100
        assert outcome["result"]["points"][POINTS[1]]["status"] == "rejected"


def test_cancel_revoke_delays_and_device_epoch(system):
    service, actor, device, link = system
    proposal = propose(service, actor, key="cancel-before")
    approve(service, actor, proposal)
    service.cancel(actor, proposal["id"], revoke=True)
    service.step()
    assert service.get(actor, proposal["id"])["status"] == "CANCELLED" and link.submissions == 0
    device.inject_fault({"ack_delay_seconds": 2, "write_delay_seconds": 2, "pv_delay_seconds": 1})
    proposal = propose(service, actor, key="cancel-after", settling_deadline_seconds=15)
    approve(service, actor, proposal)
    service.step()
    assert not service.get(actor, proposal["id"])["result"]["ack_received"]
    service.cancel(actor, proposal["id"])
    outcome = advance(service, actor, device, proposal["id"], 15)
    assert outcome["cancel_requested"] and outcome["status"] == "VERIFIED"
    assert device.read_state()["points"][POINTS[0]]["sp"] == 105
    proposal = propose(service, actor, key="epoch", targets=[{"point_id": POINTS[0], "value": 100}])
    approve(service, actor, proposal)
    service.step()
    device.inject_fault({"restart": True})
    service.step()
    assert service.get(actor, proposal["id"])["status"] == "UNKNOWN_OUTCOME"
    assert link.submissions == 2


def test_independent_mock_http_auth_and_persistent_database(tmp_path):
    root = tmp_path / "separate_device"
    app = create_mock_app(root, manual_clock=True)
    with TestClient(app, base_url="http://127.0.0.1") as http:
        assert http.get("/v1/state").status_code == 401
        driver = (root / "driver.key").read_text().strip()
        admin = (root / "test_admin.key").read_text().strip()
        client = MockClient("http://127.0.0.1", driver, admin, client=http)
        state = client.read_state()
        assert state["environment"] == "MOCK" and state["device_id"] == DEVICE
        assert http.post("/v1/test-admin/tick", json={"seconds": 1},
                         headers={"Authorization": "Bearer " + driver}).status_code == 401
        client.tick(1)
        assert client.read_state()["observation_seq"] == 1
    second = create_mock_app(root, manual_clock=True)
    with TestClient(second, base_url="http://127.0.0.1") as http:
        client = MockClient("http://127.0.0.1", driver, admin, client=http)
        new_state = client.read_state()
        assert new_state["device_epoch"] == state["device_epoch"] + 1
        assert new_state["observation_seq"] == 1



def test_workbench_api_drives_independent_mock_and_rejects_body_authority(tmp_path, monkeypatch):
    from copper_mvp.api import create_app
    monkeypatch.delenv("COPPER_MOCK_URL", raising=False)
    monkeypatch.delenv("COPPER_MOCK_KEY_FILE", raising=False)
    monkeypatch.setenv("COPPER_ASSISTANT_LIVE_CALLS", "0")
    mock_root = tmp_path / "mock"
    with TestClient(create_mock_app(mock_root, manual_clock=True), base_url="http://127.0.0.1") as device_http:
        with TestClient(create_app(run_dir=tmp_path / "workbench"), base_url="http://127.0.0.1") as workbench_http:
            wb = workbench_http.app.state.workbench
            wb.control.client = MockClient("http://127.0.0.1",
                (mock_root / "driver.key").read_text().strip(),
                (mock_root / "test_admin.key").read_text().strip(), client=device_http)
            assert workbench_http.get("/api/v2/control").status_code == 401
            workbench_http.post("/api/auth/session", json={"access_code": wb.access.owner_key_path.read_text().strip()})
            request = {"request_key": "http-flow", "reason": "合成 HTTP 端到端",
                       "targets": [{"point_id": POINTS[0], "value": 105, "unit": "A"}]}
            assert workbench_http.post("/api/v2/commands", json={**request, "role": "owner"}).status_code == 422
            assert workbench_http.post("/api/v2/commands", json={**request, "targets": [
                {"point_id": POINTS[0], "value": 105, "unit": "kA"}]}).status_code == 422
            response = workbench_http.post("/api/v2/commands", json=request)
            assert response.status_code == 201
            proposal = response.json()
            identifier = proposal["id"]
            assert workbench_http.post(f"/api/v2/commands/{identifier}/approvals",
                        json={"payload_hash": "0" * 64}).status_code == 400
            assert workbench_http.post(f"/api/v2/commands/{identifier}/approvals",
                        json={"payload_hash": proposal["payload_hash"]}).status_code == 200
            workbench_http.post(f"/api/v2/commands/{identifier}/reconcile", json={})
            for _ in range(9):
                assert workbench_http.post("/api/v2/mock/test-admin/tick", json={"seconds": 1}).status_code == 200
                result = workbench_http.get(f"/api/v2/commands/{identifier}").json()
                if result["status"] == "VERIFIED":
                    break
            assert result["status"] == "VERIFIED" and result["outbox"]["attempts"] == 1
            assert workbench_http.get(f"/api/v2/mock/devices/{DEVICE}/state").json()["points"][POINTS[0]]["sp"] == 105
            viewer = wb.access.issue(wb.access.authenticate(key=wb.access.owner_key_path.read_text().strip()), "viewer", "viewer")
            assert workbench_http.post("/api/v2/mock/test-admin/faults", json={"mode": "manual"},
                        headers={"Authorization": "Bearer " + viewer}).status_code == 403
            assert workbench_http.get(f"/api/v2/commands/{identifier}",
                        headers={"Authorization": "Bearer " + viewer}).status_code == 403
