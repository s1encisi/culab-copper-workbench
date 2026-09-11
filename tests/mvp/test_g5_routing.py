"""Routing evidence tests: delayed labels, revisions, common cohorts and switching."""
from __future__ import annotations

from datetime import datetime,timedelta,timezone
import json

import numpy as np
import pandas as pd
import pytest

from copper_mvp.common import WorkbenchError
from copper_mvp.data_contracts import LabelRecord
from copper_mvp.labels import LabelLedger
from copper_mvp.prediction_router import PolicyRouter
from copper_mvp.routing_metrics import MatureMetrics,RoutingPolicy,block_interval


def fixture(n=50):
    begin=datetime(2024,1,1,tzinfo=timezone.utc)
    labels=[]; forecasts=[]; attributes={}
    for i in range(n):
        event=f"synthetic-{i:03d}"
        decision=begin+timedelta(days=i)
        actual=[100+np.sin(i/3),1000+20*np.sin(i/4)]
        attributes[event]={"decision_at":decision.isoformat(),"mode":"A","quality":"complete","interval_group":"medium"}
        for j,(target,unit) in enumerate((("cu","g/L"),("as","mg/L"))):
            labels.append(LabelRecord(event_id=event,pair_id="pair-"+event,target=target,unit=unit,value=actual[j],
                decision_at=decision,available_at=decision+timedelta(hours=6),assumed_sample_time=decision+timedelta(hours=4),
                quality_eligible=True,source_version="synthetic-routing",source_kind="synthetic_fixture"))
        for method,error in (("Persistence",2),("Challenger",.2)):
            forecasts.append({"event_id":event,"method_id":method,"decision_at":decision.isoformat(),
                "fit_cutoff_at":(begin-timedelta(days=1)).isoformat(),"computed_at":"2026-09-11T00:00:00+00:00",
                "cu":actual[0]+error,"as":actual[1]+error,"status":"completed"})
    policy=RoutingPolicy(short_events=8,short_days=15,long_events=20,long_days=40,
        minimum_events=6,minimum_mode_events=4,minimum_blocks=4,window_stride=4,
        bootstrap_replicates=100,cooldown_labels=8)
    return pd.DataFrame(forecasts),LabelLedger(labels),attributes,policy


def engine(parts,**changes):
    forecasts,labels,attributes,policy=parts
    return MatureMetrics(changes.get("forecasts",forecasts),changes.get("labels",labels),
        changes.get("attributes",attributes),("Persistence","Challenger"),{"cu":10.,"as":100.},
        {"Persistence":{"p95_ms":.01},"Challenger":{"p95_ms":.1}},changes.get("policy",policy))


def test_future_labels_and_predictions_cannot_change_earlier_metrics_or_routes():
    parts=fixture()
    cutoff=datetime(2024,1,20,tzinfo=timezone.utc)
    original=engine(parts)
    before=original.snapshot(cutoff,"A")
    assert "synthetic-019" not in before["long"]["event_ids"]
    changed_labels=LabelLedger([r.model_copy(update={"value":r.value+1000000}) if r.available_at>cutoff else r for r in parts[1].records])
    changed_forecasts=parts[0].copy()
    future=pd.to_datetime(changed_forecasts.decision_at,utc=True)>cutoff
    changed_forecasts.loc[future,["cu","as"]]=100000000
    poisoned=engine(parts,labels=changed_labels,forecasts=changed_forecasts)
    assert poisoned.snapshot(cutoff,"A")==before
    first=PolicyRouter(original,by_mode=True)
    second=PolicyRouter(poisoned,by_mode=True)
    for event in original.events[:20]:
        assert first.decide(event)==second.decide(event)


def test_revision_recomputes_future_snapshot_without_rewriting_original():
    parts=fixture()
    source=next(r for r in parts[1].records if r.event_id=="synthetic-010" and r.target=="cu")
    changed=source.model_copy(update={"value":source.value+50,"revision":2,"supersedes":source.record_id,
        "available_at":datetime(2024,1,20,12,tzinfo=timezone.utc)})
    revised=parts[1].with_revision(changed)
    original=engine(parts)
    newer=engine(parts,labels=revised)
    early=datetime(2024,1,20,tzinfo=timezone.utc)
    saved=original.snapshot(early,"A")
    serialized=json.dumps(saved,sort_keys=True)
    assert newer.snapshot(early,"A")==saved
    late=early+timedelta(days=1)
    first,second=original.snapshot(late,"A"),newer.snapshot(late,"A")
    assert first["long"]["label_revision_hash"]!=second["long"]["label_revision_hash"]
    assert first["long"]["metrics"]["Persistence"]["cu"]["mae"]!=second["long"]["metrics"]["Persistence"]["cu"]["mae"]
    assert json.dumps(saved,sort_keys=True)==serialized


def test_common_window_counts_missing_predictions_without_extending_it():
    parts=fixture()
    data=parts[0].copy()
    missing=data.method_id.eq("Challenger") & data.event_id.eq("synthetic-017")
    data.loc[missing,["cu","as"]]=np.nan
    data.loc[missing,"status"]="failed"
    policy=parts[3].model_copy(update={"minimum_coverage":.8})
    state=engine(parts,forecasts=data,policy=policy).snapshot(datetime(2024,1,21,tzinfo=timezone.utc),"A")
    short=state["short"]
    assert short["coverage"]["Challenger"]["missing_predictions"]==1
    assert short["coverage"]["Challenger"]["eligible_events"]==8
    assert short["n"]==7 and "synthetic-012" in short["event_ids"] and "synthetic-011" not in short["event_ids"]
    assert short["metrics"]["Persistence"]["cu"]["n"]==short["metrics"]["Challenger"]["cu"]["n"]==7
    with pytest.raises(WorkbenchError,match="完整"):
        engine(parts,forecasts=data[~missing])
    unsupported=engine(parts).snapshot(datetime(2024,1,21,tzinfo=timezone.utc),"unseen-mode")
    assert unsupported["status"]=="INSUFFICIENT_LABELS"


def test_switch_uses_distinct_mature_windows_and_invalid_prediction_falls_back():
    parts=fixture()
    data=parts[0].copy()
    data.loc[data.method_id.eq("Challenger") & data.event_id.eq("synthetic-016"),["cu","as"]]=np.nan
    router=PolicyRouter(engine(parts,forecasts=data),by_mode=False)
    results=[]
    for event in router.metrics.events[:17]:
        results.append(router.decide(event))
    switches=[(i,r["cu"]) for i,r in enumerate(results) if r["cu"]["reason"]=="SHADOW_SWITCH"]
    assert switches and switches[0][0]>=12
    assert results[8]["cu"]["reason"]=="WAIT_CONFIRMATION"
    assert results[8]["cu"]["confirmation_windows"]==1
    assert results[9]["cu"]["confirmation_windows"]==1
    assert results[16]["cu"]["selected_method"]=="Persistence"
    assert results[16]["cu"]["reason"]=="CURRENT_PREDICTION_INVALID"
    assert all(not d["live_model_change"] for r in results for d in r.values())
    for r in results:
        for d in r.values():
            assert datetime.fromisoformat(d["metric_as_of"])<=datetime.fromisoformat(d["decision_at"])


def test_future_fit_and_insufficient_blocks_are_explicit():
    parts=fixture()
    data=parts[0].copy()
    data.loc[0,"fit_cutoff_at"]="2024-01-03T00:00:00+00:00"
    with pytest.raises(WorkbenchError,match="时间"):
        engine(parts,forecasts=data)
    dates=[datetime(2024,1,1,tzinfo=timezone.utc)+timedelta(days=i) for i in range(8)]
    value=block_interval(np.arange(8),dates,RoutingPolicy(minimum_blocks=20))
    assert value["status"]=="INSUFFICIENT_BLOCKS" and value["high"] is None
    original=engine(parts)
    original.profiles["Challenger"]["p95_ms"]=1000
    state=original.snapshot(datetime(2024,1,21,tzinfo=timezone.utc),"A")
    assert state["long"]["excluded"]["Challenger"]=="INFERENCE_BUDGET"


def test_confirming_evidence_is_retained_at_switch_and_cooldown():
    parts=fixture(65)
    data=parts[0].copy()
    later=data.method_id.eq("Challenger") & (data.event_id>="synthetic-020")
    data.loc[later,["cu","as"]]+=3.8
    policy=parts[3].model_copy(update={"cooldown_days":100,"cooldown_labels":100})
    snapshots={}
    router=PolicyRouter(engine(parts,forecasts=data,policy=policy),by_mode=False,
        on_snapshot=lambda value:snapshots.update({value["id"]:value}))
    values=[router.decide(e)["cu"] for e in router.metrics.events]
    switched=next(d for d in values if d["reason"]=="SHADOW_SWITCH")
    assert switched["confirmation_windows"]==2
    assert len(set(switched["confirmation_snapshots"]))==2
    assert all(ref in snapshots for ref in switched["confirmation_snapshots"])
    assert all(snapshots[ref]["as_of"]<=switched["decision_at"] for ref in switched["confirmation_snapshots"])
    cooldown=[d for d in values if d["reason"]=="COOLDOWN"]
    assert cooldown and all(d["selected_method"]=="Challenger" for d in cooldown)


def test_routing_api_authorization_and_evidence_integrity(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from copper_mvp.api import create_app
    from copper_mvp.common import digest,file_hash,write_json
    from copper_mvp.routing_studies import RoutingStudies
    monkeypatch.delenv("COPPER_MOCK_URL",raising=False)
    monkeypatch.delenv("COPPER_MOCK_KEY_FILE",raising=False)
    monkeypatch.setenv("COPPER_ASSISTANT_LIVE_CALLS","0")
    with TestClient(create_app(run_dir=tmp_path/"runtime"),base_url="http://127.0.0.1") as http:
        wb=http.app.state.workbench
        studies=RoutingStudies(tmp_path/"studies",wb.data)
        wb._routing_studies=studies
        identifier="a"*32
        root=studies.directory(identifier)
        root.mkdir()
        snapshot={"schema_version":"synthetic","as_of":"2024-01-20T00:00:00+00:00","n":6}
        reference=digest(snapshot)
        write_json(root/"metric_snapshots"/(reference+".json"),{"id":reference,**snapshot})
        write_json(root/"protocol.json",{"scope":"synthetic authorization fixture"})
        (root/"predictions.csv").write_bytes(b"event_id,cu,as\nsynthetic,100,1000\n")
        decision={"event_id":"synthetic","metric_snapshot_id":reference,"confirmation_snapshots":[reference]}
        (root/"decisions.jsonl").write_bytes((json.dumps(decision)+"\n").encode("utf-8"))
        manifest={"protocol_sha256":file_hash(root/"protocol.json"),"predictions_sha256":file_hash(root/"predictions.csv"),
                  "decisions_sha256":file_hash(root/"decisions.jsonl")}
        write_json(root/"replay_manifest.json",manifest)
        write_json(root/"evaluation.json",{"scope":"synthetic","replay_manifest_sha256":file_hash(root/"replay_manifest.json")})
        write_json(root/"state.json",{"id":identifier,"status":"completed","owner_id":"owner","project_id":"copper-research",
            "created_at":"2024-01-20T00:00:00+00:00","evaluation_sha256":file_hash(root/"evaluation.json")})
        assert http.get("/api/v2/prediction-studies").status_code==401
        owner=wb.access.authenticate(key=wb.access.owner_key_path.read_text().strip())
        http.post("/api/auth/session",json={"access_code":wb.access.owner_key_path.read_text().strip()})
        assert http.get("/api/v2/prediction-studies").json()["items"][0]["id"]==identifier
        assert http.get(f"/api/v2/prediction-studies/{identifier}/decisions",params={"event_id":"synthetic"}).status_code==200
        key=wb.access.issue(owner,"viewer","viewer")
        headers={"Authorization":"Bearer "+key}
        assert http.get("/api/v2/prediction-studies",headers=headers).json()["items"]==[]
        assert http.get(f"/api/v2/prediction-studies/{identifier}",headers=headers).status_code==403
        assert http.post("/api/v2/prediction-studies",headers=headers,json={"request_key":"denied","comparison_id":"b"*32}).status_code==403
        write_json(root/"metric_snapshots"/(reference+".json"),{"id":reference,**snapshot,"n":999})
        assert http.get(f"/api/v2/prediction-studies/{identifier}/decisions",params={"event_id":"synthetic"}).status_code==400
        (root/"predictions.csv").write_bytes(b"changed prediction")
        assert http.get(f"/api/v2/prediction-studies/{identifier}").status_code==400
