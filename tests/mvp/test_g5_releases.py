"""Release invariants with synthetic source outcomes and actual access-key checks."""
from datetime import datetime,timezone
import json
import pytest

from copper_mvp.access import AccessControl,Principal
from copper_mvp.common import WorkbenchError,digest
from copper_mvp.model_lifecycle import ModelLifecycle
from copper_mvp.release_contracts import TASK_ID,BASELINE_ID
from copper_mvp.research_store import ResearchStore
from copper_mvp.storage import RunStore


class Source:
    broken=False
    improvement=.6
    def load(self,actor,request):
        d={"method_id":request.method_id,"target":request.target,"unit":"g/L" if request.target=="cu" else "mg/L","task_id":TASK_ID,
           "benchmark":{"n":100,"coverage":1.,"mae":1-self.improvement,"baseline_mae":1.,
             "paired_ci95":{"low":-.7,"high":-.5},"negative_predictions":0,"p95_ms":1.,
             "mode_metrics":{"A":{"n":100,"mae_mean_over_seeds":1-self.improvement}},
             "baseline_modes":{"A":{"n":100,"mae_mean_over_seeds":1.}}},
           "proxy_approved":False,"causal_control":False,"scopes":["historical_oof"],"fixture":True}
        return {**d,"source_fingerprint":digest(d)}
    def verify(self,descriptor):
        if self.broken:raise WorkbenchError("synthetic artifact changed","MODEL_HASH_MISMATCH")
        return True
    def probe(self,descriptor,samples):
        self.verify(descriptor)
        return {"parity_passed":True,"samples":samples*5,"scope":"historical_runtime_shadow","independent_new_labels":False}
    def event(self,event):
        return {"event_id":event,"decision_at":"2024-01-01T12:00:00+00:00","snapshot_id":"synthetic-snapshot",
                "dataset_version":"synthetic","current":{"cu":10.,"as":1000.}}
    def predict(self,descriptor,event):
        self.verify(descriptor)
        return (11. if descriptor["target"]=="cu" else 1100.),{"artifact_sha256":"synthetic","execution":"synthetic_model"}


@pytest.fixture
def system(tmp_path):
    store=RunStore(tmp_path)
    access=AccessControl(ResearchStore(store))
    owner=access.authenticate(key=access.owner_key_path.read_text().strip())
    clock=[datetime(2024,1,2,tzinfo=timezone.utc).timestamp()]
    source=Source()
    service=ModelLifecycle(store,access,source,clock=lambda:clock[0])
    return service,owner,source,clock


def ready(service,actor,key="artifact"):
    artifact=service.register(actor,{"request_key":key,"source_kind":"model_comparison","source_id":"a"*32,
                           "method_id":"SyntheticLinear","target":"cu"})
    shadow=service.start_shadow(actor,artifact["id"],{"request_key":key+"-shadow"})
    return service.artifact(actor,artifact["id"]),shadow


def proposal(service,actor,artifact,shadow,key="release",version=0):
    return service.propose(actor,{"request_key":key,"candidate_id":artifact["id"],"shadow_id":shadow["id"],
                          "target":"cu","expected_version":version,"reason":"synthetic workflow verification"})


def approve(service,actor,value):
    return service.approve(actor,value["id"],{"payload_hash":value["payload_hash"]})


def test_release_changes_new_predictions_and_rollback_preserves_old_records(system):
    service,owner,source,clock=system
    before=service.forecast(owner,{"request_key":"before","event_id":"event"})
    assert before["predictions"]["cu"]["model"]=="Persistence"
    artifact,shadow=ready(service,owner)
    release=proposal(service,owner,artifact,shadow)
    approved=approve(service,owner,release)
    assert approved["status"]=="applied"
    after=service.forecast(owner,{"request_key":"after","event_id":"event"})
    assert after["predictions"]["cu"]["value"]==11
    assert after["predictions"]["cu"]["release_id"]==release["id"]
    historical=service.forecast(owner,{"request_key":"historical","event_id":"event","selection":"as_of_event"})
    assert historical["predictions"]["cu"]["value"]==10  # Approval did not exist at the event time.
    rollback=service.rollback(owner,{"request_key":"rollback","target":"cu","expected_version":1,"reason":"synthetic rollback"})
    approve(service,owner,rollback)
    final=service.forecast(owner,{"request_key":"final","event_id":"event"})
    assert final["predictions"]["cu"]["value"]==10
    assert service.prediction(owner,after["id"])==after
    assert service.forecast(owner,{"request_key":"after","event_id":"event"})==after
    assert service.pointers(owner)["cu"]["version"]==2


def test_exact_approval_version_permission_and_source_are_rechecked(system):
    service,owner,source,clock=system
    artifact,shadow=ready(service,owner)
    one=proposal(service,owner,artifact,shadow,key="one")
    two=proposal(service,owner,artifact,shadow,key="two")
    with pytest.raises(WorkbenchError,match="哈希"):
        service.approve(owner,one["id"],{"payload_hash":"0"*64})
    with pytest.raises(WorkbenchError,match="权限"):
        service.approve(Principal("viewer","viewer"),one["id"],{"payload_hash":one["payload_hash"]})
    approve(service,owner,one)
    with pytest.raises(WorkbenchError,match="指针"):
        approve(service,owner,two)
    with pytest.raises(WorkbenchError,match="回退"):
        service.retire(owner,artifact["id"],service.artifact(owner,artifact["id"])["version"])
    source.broken=True
    output=service.forecast(owner,{"request_key":"broken","event_id":"event"})
    assert output["predictions"]["cu"]["model"]=="Persistence"
    assert output["predictions"]["cu"]["fallback_reason"]=="MODEL_HASH_MISMATCH"
    assert service.artifact(owner,artifact["id"])["status"]=="quarantined"


def test_small_gains_or_proxy_request_do_not_publish(system):
    service,owner,source,clock=system
    source.improvement=.01
    artifact,shadow=ready(service,owner)
    assert "MINIMUM_IMPROVEMENT" in service.qualification(owner,artifact["id"],shadow["id"])["failures"]
    with pytest.raises(WorkbenchError,match="发布条件"):
        proposal(service,owner,artifact,shadow)
    source.improvement=.6
    other,other_shadow=ready(service,owner,key="proxy")
    with pytest.raises(WorkbenchError,match="PROXY_QUALIFICATION_REQUIRED"):
        service.propose(owner,{"request_key":"proxy-release","candidate_id":other["id"],"shadow_id":other_shadow["id"],
            "target":"cu","purpose":"scenario_prediction","expected_version":0,"reason":"synthetic denial"})
    assert service.pointers(owner)["cu"]["candidate_id"]==BASELINE_ID


def test_expired_revoked_and_tampered_evidence_cannot_apply(system):
    service,owner,source,clock=system
    key=service.access.issue(owner,"author","researcher")
    author=service.access.authenticate(key=key)
    artifact,shadow=ready(service,author)
    release=proposal(service,author,artifact,shadow)
    service.access.revoke(owner,key)
    with pytest.raises(WorkbenchError,match="授权"):
        approve(service,owner,release)
    artifact,shadow=ready(service,owner,key="expires")
    release=proposal(service,owner,artifact,shadow,key="expires-release")
    clock[0]+=1000
    with pytest.raises(WorkbenchError,match="过期"):
        approve(service,owner,release)
    current=proposal(service,owner,artifact,shadow,key="tampered")
    with service.connection() as c:
        c.execute("UPDATE model_shadows SET evidence=? WHERE id=?",(json.dumps({"parity_passed":False}),shadow["id"]))
    with pytest.raises(WorkbenchError):
        approve(service,owner,current)
    assert service.pointers(owner)["cu"]["candidate_id"]==BASELINE_ID


def test_reject_can_close_expired_proposal_and_preserves_applied_release(system):
    service,owner,source,clock=system
    artifact,shadow=ready(service,owner)
    release=proposal(service,owner,artifact,shadow)
    clock[0]+=1000
    closed=service.approve(owner,release["id"],{"payload_hash":release["payload_hash"],"decision":"reject","reason":"expired"})
    assert closed["status"]=="rejected" and service.pointers(owner)["cu"]["version"]==0


def test_release_api_identity_and_exact_pointer_flow(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from copper_mvp.api import create_app
    monkeypatch.delenv("COPPER_MOCK_URL",raising=False)
    monkeypatch.delenv("COPPER_MOCK_KEY_FILE",raising=False)
    monkeypatch.setenv("COPPER_ASSISTANT_LIVE_CALLS","0")
    with TestClient(create_app(run_dir=tmp_path/"api"),base_url="http://127.0.0.1") as client:
        wb=client.app.state.workbench
        wb._model_lifecycle=ModelLifecycle(wb.store,wb.access,Source())
        assert client.get("/api/v2/model-pointers").status_code==401
        key=wb.access.owner_key_path.read_text().strip()
        client.post("/api/auth/session",json={"access_code":key})
        data={"request_key":"api-artifact","source_kind":"model_comparison","source_id":"b"*32,"method_id":"SyntheticLinear","target":"cu"}
        assert client.post("/api/v2/model-artifacts",json={**data,"role":"owner"}).status_code==422
        registered=client.post("/api/v2/model-artifacts",json=data).json()
        owner=wb.access.authenticate(key=key)
        shadow=wb._model_lifecycle.start_shadow(owner,registered["id"],{"request_key":"api-shadow"})
        proposed=client.post("/api/v2/releases",json={"request_key":"api-release","candidate_id":registered["id"],"shadow_id":shadow["id"],
                "target":"cu","expected_version":0,"reason":"synthetic API check"})
        assert proposed.status_code==201
        release=proposed.json()
        viewer=wb.access.issue(owner,"viewer","viewer")
        headers={"Authorization":"Bearer "+viewer}
        assert client.post("/api/v2/releases/"+release["id"]+"/approve",json={"payload_hash":release["payload_hash"]},headers=headers).status_code==403
        assert client.post("/api/v2/releases/"+release["id"]+"/approve",json={"payload_hash":release["payload_hash"]}).json()["status"]=="applied"
        prediction=client.post("/api/v2/release-predictions",json={"request_key":"api-prediction","event_id":"synthetic-event"}).json()
        assert prediction["predictions"]["cu"]["value"]==11
        rollback=client.post("/api/v2/release-rollbacks",json={"request_key":"api-rollback","target":"cu","expected_version":1,"reason":"synthetic rollback"}).json()
        assert client.post("/api/v2/releases/"+rollback["id"]+"/approve",json={"payload_hash":rollback["payload_hash"]}).json()["status"]=="applied"
        assert client.get("/api/v2/release-predictions/"+prediction["id"]).json()==prediction
        assert client.get("/api/v2/release-predictions/"+prediction["id"],headers=headers).status_code==403


def test_missing_current_value_is_explicit_and_does_not_invent_prediction(system):
    service,owner,source,clock=system
    original=source.event
    source.event=lambda event:{**original(event),"current":{"cu":None,"as":1000.}}
    result=service.forecast(owner,{"request_key":"missing","event_id":"event"})
    assert result["predictions"]["cu"]["value"] is None
    assert result["predictions"]["cu"]["fallback_reason"]=="CURRENT_RESULT_MISSING_OR_INVALID"
    assert result["predictions"]["as"]["value"]==1000
    assert service.prediction(owner,result["id"])==result


def test_prediction_integrity_is_checked_on_read_and_idempotent_reuse(system):
    service,owner,source,clock=system
    output=service.forecast(owner,{"request_key":"integrity","event_id":"event"})
    altered={**output,"predictions":{**output["predictions"],"cu":{**output["predictions"]["cu"],"value":999.}}}
    with service.connection() as c:
        c.execute("UPDATE release_predictions SET result=? WHERE id=?",(json.dumps(altered),output["id"]))
    with pytest.raises(WorkbenchError,match="哈希"):
        service.prediction(owner,output["id"])
    with pytest.raises(WorkbenchError,match="哈希"):
        service.forecast(owner,{"request_key":"integrity","event_id":"event"})
