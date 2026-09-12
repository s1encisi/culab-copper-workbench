"""Statistical adapter checks with preserved raw event positions and causal updates."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from copper_mvp.common import WorkbenchError
from copper_mvp.data_contracts import LabelRecord
from copper_mvp.labels import LabelLedger
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_registry import ComparisonRequest,ComparisonPredictionRequest
from copper_mvp.statistical_registry import STATISTICAL_METHODS,SEQUENCE_METHODS


class SeriesData:
    def __init__(self,root):
        root.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(17);n=160
        self.ids=[f"event-{i:03d}" for i in range(n)]
        times=[datetime(2024,1,1,tzinfo=timezone.utc)]
        for i in range(1,n):times.append(times[-1]+timedelta(hours=6+(i%3)*2))
        self.feature_columns=["origin_cu_g_l","origin_as_mg_l","stage3_current_a__t_minus_0h","stage4_current_a__t_minus_0h"]+[f"x{i}" for i in range(4,114)]
        X=rng.normal(size=(n,114));X[:,2:4]=100+10*rng.normal(size=(n,2));X[:,110:]=rng.integers(0,3,(n,4))
        X[0,:2]=[40,1000]
        for i in range(1,n):
            X[i,:2]=np.array([40,1000])+.85*(X[i-1,:2]-[40,1000])+(X[i-1,2:4]-100)*[.01,.3]+rng.normal(size=2)*[.05,2.]
        self.X=pd.DataFrame(X,index=self.ids,columns=self.feature_columns)
        quality=np.ones(n,dtype=bool);quality[[20,35,65]]=False
        self.frame=pd.DataFrame({"origin_event_id":self.ids,"decision_at":times,"origin_result_quality_eligible":quality,
            "stage3_current_a__t_minus_0h__age_h":np.zeros(n),"stage4_current_a__t_minus_0h__age_h":np.zeros(n)},index=self.ids)
        self.admissions={e:{"admission_status":"ADMITTED","feature_cutoff_at":t.isoformat()} for e,t in zip(self.ids,times)}
        self.mode_cards={e:{"mode_code":"synthetic"} for e in self.ids}
        self.primary_ids={self.ids[i] for i in range(n-1) if quality[i] and quality[i+1]}
        self.dataset_version="synthetic-series"
        records=[]
        for i,e in enumerate(self.ids[:-1]):
            for target,unit,j in (("cu","g/L",0),("as","mg/L",1)):
                records.append(LabelRecord(event_id=e,pair_id="pair-"+e,target=target,unit=unit,value=float(X[i+1,j]),
                    decision_at=times[i],available_at=times[i+1],assumed_sample_time=times[i+1]-timedelta(hours=2),
                    quality_eligible=e in self.primary_ids,source_version="synthetic",source_kind="synthetic_fixture"))
        self.ledger=LabelLedger(records)
        self.fold_for={e:"FOLD_1" for e in self.ids[60:100] if e in self.primary_ids}
        train=[e for e in self.ids[:60] if e in self.primary_ids]
        self.cv=pd.DataFrame([{"origin_event_id":e,"fold_id":"FOLD_1","fold_role":"TRAIN" if e in train else "VALIDATION",
            "fold_fit_cutoff_at":times[60].isoformat()} for e in train+list(self.fold_for)])
        sources={}
        for name in ("features","admissions","index","folds","feature_contract","timing_contract","labels","pairing"):
            path=root/(name+".json");path.write_text(json.dumps({"source":name}),encoding="utf-8");sources[name]=path
        self.paths=SimpleNamespace(sources=lambda:sources)

    def row(self,event):return self.frame.loc[event]
    def training_ids(self,fold):
        return [e for e in self.ids if e in self.primary_ids] if fold is None else self.cv.loc[self.cv.fold_role.eq("TRAIN"),"origin_event_id"].tolist()


@pytest.mark.parametrize("method",SEQUENCE_METHODS)
def test_sequence_keeps_positions_and_future_changes_do_not_leak(method,tmp_path):
    data=SeriesData(tmp_path/"inputs");cutoff=data.row("event-060").decision_at
    model=RegisteredModel(method).fit_context(data,data.training_ids("FOLD_1"),cutoff)
    assert model.fit_metadata["raw_series_records"]==61
    assert model.fit_metadata["missing_per_target"]==[2,2]
    assert model.fit_metadata["original_event_index_preserved"]
    params=[m.params.copy() for m in model.models]
    events=["event-060","event-070","event-080"]
    before=model.predict_context(data,events)
    assert before.shape==(3,2) and np.isfinite(before).all()
    assert np.allclose(model.predict_context(data,["event-070"])[0],before[1])
    data.X.loc[data.ids[90:],data.feature_columns[:4]]+=1000
    assert np.allclose(model.predict_context(data,events),before)
    assert all(np.array_equal(m.params,p) for m,p in zip(model.models,params))
    uncertainty=model.predict_uncertainty_context(data,events)
    assert np.isfinite(uncertainty["std"]).all() and not uncertainty["calibrated"]
    path=tmp_path/(method+".joblib");joblib.dump(model,path)
    assert np.allclose(joblib.load(path).predict_context(data,events),before)
    with pytest.raises(WorkbenchError,match="之后"):
        model.predict_context(data,["event-050"])
    data.X.loc["event-010",data.feature_columns[0]]+=1
    with pytest.raises(WorkbenchError,match="改变"):
        model.predict_context(data,events)


@pytest.mark.parametrize("method",("LocalLinearKernel","PSplineGAM"))
def test_statistical_regressors_fit_and_serialize(method,tmp_path):
    data=SeriesData(tmp_path/"inputs");ids=data.training_ids("FOLD_1")
    X=data.X.loc[ids].to_numpy()
    labels=data.ledger.latest(data.row("event-060").decision_at)
    y=np.array([[labels[(e,t)].value for t in ("cu","as")] for e in ids])
    model=RegisteredModel(method).fit(X,y)
    probe=data.X.loc[data.ids[70:75]].to_numpy()
    predicted=model.predict(probe)
    assert predicted.shape==(5,2) and np.isfinite(predicted).all()
    path=tmp_path/(method+".joblib");joblib.dump(model,path)
    assert np.allclose(joblib.load(path).predict(probe),predicted)


def test_contextual_training_comparison_and_api_prediction(tmp_path,monkeypatch):
    from copper_mvp.model_training import train_comparison
    from copper_mvp.model_evaluation import evaluate_comparison
    from copper_mvp.model_comparisons import ComparisonService
    from copper_mvp.common import write_json,file_hash
    data=SeriesData(tmp_path/"inputs")
    class Labels:
        def __init__(self,data):self.labels=data.ledger
        def snapshot(self,event):return {"id":"synthetic-"+event}
    for module in ("model_training","model_evaluation","model_comparisons"):
        monkeypatch.setattr("copper_mvp."+module+".DataService",Labels)
    root=tmp_path/("a"*32)
    request=ComparisonRequest(request_key="sequence",methods=("Persistence","VAR"))
    manifest=train_comparison(data,request,root)
    evaluated=evaluate_comparison(data,root)
    assert evaluated["common_events"]==len(data.fold_for)
    artifact=next(v for v in manifest["artifacts"] if v["method_id"]=="VAR" and v["fold_id"]=="FOLD_1")
    assert artifact["fit_metadata"]["raw_series_records"]==61
    write_json(root/"state.json",{"run_id":root.name,"status":"completed","created_at":"synthetic","fingerprint":"synthetic",
        "request":request.model_dump(mode="json"),"evaluation_sha256":file_hash(root/"evaluation.json")})
    result=ComparisonService(tmp_path,data).predict(root.name,ComparisonPredictionRequest(event_id="event-070",method_id="VAR",include_uncertainty=True))
    assert result["uncertainty"]["state_updates_only"] and len(result["uncertainty"]["std"][0])==2


def test_sarimax_uses_current_exog_and_missing_event_positions(tmp_path):
    data=SeriesData(tmp_path/"inputs")
    model=RegisteredModel("SARIMAX").fit_context(data,data.training_ids("FOLD_1"),data.row("event-060").decision_at)
    before=model.predict_context(data,["event-070"]).copy()
    data.X.loc["event-071",data.feature_columns[:4]]+=1000
    assert np.allclose(model.predict_context(data,["event-070"]),before)
    data.X.loc["event-070","stage3_current_a__t_minus_0h"]+=20
    assert not np.allclose(model.predict_context(data,["event-070"]),before)
    missing=model.predict_uncertainty_context(data,["event-065"])
    assert np.isfinite(missing["mean"]).all() and np.isfinite(missing["std"]).all()


def test_ets_missing_update_keeps_trained_noise_and_parameters(tmp_path):
    data=SeriesData(tmp_path/"inputs")
    model=RegisteredModel("ETS").fit_context(data,data.training_ids("FOLD_1"),data.row("event-060").decision_at)
    parameters=[r.params.copy() for r in model.models]
    result=model.predict_uncertainty_context(data,["event-064","event-065","event-066"])
    assert np.isfinite(result["std"]).all() and (result["std"]>0).all()
    assert all(np.array_equal(a,r.params) for a,r in zip(parameters,model.models))


@pytest.mark.parametrize("method",("SARIMAX","ETS"))
def test_context_predictions_equal_separate_fixed_parameter_prefix_filters(method,tmp_path):
    from copper_mvp.statistical_models import event_arrays
    data=SeriesData(tmp_path/"inputs")
    model=RegisteredModel(method).fit_context(data,data.training_ids("FOLD_1"),data.row("event-060").decision_at)
    events=["event-060","event-065","event-078"]
    actual=model.predict_context(data,events)
    seq=model._sequence;ids,times,Y,exog=event_arrays(data,method=="SARIMAX")
    for i,event in enumerate(events):
        index=ids.index(event);end=index+1
        normalized=(Y[:end]-seq.mean_)/seq.scale_
        regressors=None;extra={}
        if method=="SARIMAX":
            lagged=np.vstack((np.full((1,2),np.nan),exog[:end-1]))
            regressors=np.column_stack((np.ones(end),seq.exog_scale.transform(seq.exog_imputer.transform(lagged))))
            extra["exog"]=np.column_stack((np.ones(1),seq.exog_scale.transform(seq.exog_imputer.transform(exog[end-1:end]))))
        expected=[]
        for t,result in enumerate(seq.models):
            cloned=result.model.clone(normalized[:,t],exog=regressors) if regressors is not None else result.model.clone(normalized[:,t])
            filtered=cloned.filter(result.params)
            expected.append(float(np.asarray(filtered.get_forecast(steps=1,**extra).predicted_mean)[0])*seq.scale_[t]+seq.mean_[t])
        assert np.allclose(actual[i],expected,rtol=1e-10,atol=1e-9)
