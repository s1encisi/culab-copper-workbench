"""Public synthetic replay: evidence binding, exact approval and durable Mock feedback."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from copper_mvp.access import AccessControl, Principal
from copper_mvp.common import WorkbenchError, digest
from copper_mvp.control.commands import CommandService
from copper_mvp.control.contracts import POINTS
from copper_mvp.control.mock import MockDevice
from copper_mvp.research_store import ResearchStore
from copper_mvp.research_tools import public_evidence, render_answer
from copper_mvp.storage import RunStore


class DemoDeviceLink:
    """In-process transport to an independently persisted synthetic device."""

    admin_key = True

    def __init__(self, device):
        self.device = device
        self.submissions = 0

    def __getattr__(self, name):
        return getattr(self.device, name)

    def close(self):
        pass

    def submit_command(self, *args):
        self.submissions += 1
        result = self.device.submit_command(*args)
        if result["ack_lost"]:
            raise WorkbenchError("Synthetic ACK loss", "MOCK_TRANSPORT")
        return {**result, "ack_received": result["ack_at"] <= self.device.read_state()["virtual_time"]}


def scenario(root, ack_loss):
    store = RunStore(root / "workbench")
    access = AccessControl(ResearchStore(store))
    owner = access.authenticate(key=access.owner_key_path.read_text(encoding="utf-8").strip())
    device = MockDevice(root / "device")
    if ack_loss:
        device.inject_fault({"ack_loss": True})
    link = DemoDeviceLink(device)
    service = CommandService(store, access, link, autostart=False)
    try:
        request = {
            "request_key": "public-demo",
            "reason": "Public synthetic 100 A to 105 A example",
            "targets": [{"point_id": POINTS[0], "value": 105, "unit": "A"}],
        }
        proposal = service.propose(owner, request)
        idempotent = service.propose(owner, request)["id"] == proposal["id"]
        service.step()
        no_write_before_approval = link.submissions == 0
        blocked = {}
        for label, actor, hash_value in [
            ("researcher_cannot_approve", Principal("demo-researcher", "researcher"), proposal["payload_hash"]),
            ("changed_payload_rejected", owner, "0" * 64),
        ]:
            try:
                service.approve(actor, proposal["id"], {"payload_hash": hash_value})
            except WorkbenchError as error:
                blocked[label] = error.code
            else:
                raise AssertionError(label)
        service.approve(owner, proposal["id"], {"payload_hash": proposal["payload_hash"]})
        service.step()
        states = [service.get(owner, proposal["id"])["status"]]
        for _ in range(20):
            device.tick(1)
            service.step()
            result = service.get(owner, proposal["id"])
            if result["status"] != states[-1]:
                states.append(result["status"])
            if result["status"] in ("VERIFIED", "FAILED", "PARTIAL", "REJECTED"):
                break
        device_record = device.get_command_status(proposal["id"])
        checks = {
            "no_write_before_approval": no_write_before_approval,
            "idempotent_proposal": idempotent,
            "exact_approval_and_role_checked": len(blocked) == 2,
            "verified_feedback": result["status"] == "VERIFIED",
            "one_device_write": device_record["write_count"] == 1,
            "one_delivery_attempt": result["outbox"]["attempts"] == 1,
            "three_good_samples": result["result"]["points"][POINTS[0]]["good_samples"] == 3,
        }
        if not all(checks.values()):
            raise AssertionError(checks)
        return {
            "scenario": "ack_loss" if ack_loss else "normal",
            "states": states,
            "checks": checks,
            "final_status": result["status"],
            "device_write_count": device_record["write_count"],
            "ack_received": result["result"]["ack_received"],
            "blocked_actions": blocked,
        }
    finally:
        service.close()


def run(output):
    root = Path(output) / uuid.uuid4().hex[:12]
    root.mkdir(parents=True)
    evidence_id = "E-" + digest({"demo": "public-facts-v1"})[:20]
    evidence = {
        "evidence_id": evidence_id,
        "tool": "event_context",
        "data": {
            "summary": {"dataset": "synthetic", "local_fact_names": ["current_cu", "current_as"]},
            "local_facts": {
                "current_cu": {"value": 12.5, "unit": "g/L"},
                "current_as": {"value": 1000.0, "unit": "mg/L"},
            },
        },
    }
    model_visible = public_evidence(evidence)
    assert "local_facts" not in model_visible["data"]
    template = "Cu={{" + evidence_id + ".current_cu}}; As={{" + evidence_id + ".current_as}}."
    answer = render_answer(template, [evidence])
    assert "12.5 g/L" in answer and "1000 mg/L" in answer
    results = [scenario(root / "normal", False), scenario(root / "ack_loss", True)]
    report = {
        "schema_version": "public-demo.v1",
        "dataset": "synthetic",
        "decision_mode": "deterministic replay",
        "paid_model_calls": 0,
        "real_plant_connected": False,
        "answer": answer,
        "model_visible_evidence": model_visible,
        "scenarios": results,
        "passed": True,
    }
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CuLab public demo",
        "",
        "Synthetic inputs and deterministic decision replay; actual evidence rendering and control services.",
        "No provider key or private plant dataset is used.",
        "",
        "| Scenario | Final status | Device writes | ACK received |",
        "| --- | --- | ---: | --- |",
    ]
    lines += [
        "| "
        + r["scenario"]
        + " | "
        + r["final_status"]
        + " | "
        + str(r["device_write_count"])
        + " | "
        + str(r["ack_received"])
        + " |"
        for r in results
    ]
    lines += [
        "",
        (
            "Checks: exact approval, role enforcement, idempotency, three feedb"
            "ack samples and ACK-loss reconciliation all passed."
        ),
        "",
        "The LLM decision is replayed in this offline example. The production research loop is in research_service.py.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"passed": True, "scenarios": len(results), "provider_calls": 0, "report": str(root / "report.md")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/public-demo"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False))
