"""G6 method contracts: real fits, serialization, weights and train-only transforms."""
import joblib
import numpy as np
import pytest

from copper_mvp.common import WorkbenchError
from copper_mvp.classical_registry import CLASSICAL_METHODS,WEIGHTED
from copper_mvp.classical_models import StableNumericScale
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_registry import METHOD_IDS,LEGACY_METHOD_IDS,ComparisonRequest,catalog


def data(n=96):
    rng=np.random.default_rng(42)
    X=rng.normal(size=(n,114))
    X[:,0]+=20
    X[:,1]=1000+20*X[:,1]
    X[:,110:]=rng.integers(0,3,(n,4))
    X[::7,4]=np.nan
    X[:,7]=rng.normal(scale=1e-15,size=n)
    y=X[:,:2]+np.column_stack((.2*X[:,2],2*X[:,3]))+rng.normal(0,.02,(n,2))
    return X,y


def test_registry_counts_mechanisms_and_preserves_default_scope():
    assert len(CLASSICAL_METHODS)==14 and len(METHOD_IDS)==20
    assert catalog()["new_method_count"]==17
    assert ComparisonRequest(request_key="default").methods==LEGACY_METHOD_IDS
    assert len({item["implementation"] for item in catalog()["items"]})==20


@pytest.mark.parametrize("method",CLASSICAL_METHODS)
def test_each_new_method_fits_and_roundtrips(method,tmp_path):
    X,y=data()
    model=RegisteredModel(method).fit(X[:72],y[:72])
    predicted=model.predict(X[72:])
    assert predicted.shape==(24,2) and np.isfinite(predicted).all()
    path=tmp_path/(method+".joblib")
    joblib.dump(model,path)
    assert np.allclose(joblib.load(path).predict(X[72:]),predicted,rtol=1e-12,atol=1e-12)


def test_training_noise_feature_is_inactive_without_using_future_range():
    X,y=data()
    scale=StableNumericScale().fit(X[:72,[0,7]])
    assert scale.active_.tolist()==[True,False]
    initial=scale.mean_.copy()
    values=X[72:,[0,7]].copy();values[:,1]=10000
    assert np.all(scale.transform(values)[:,1]==0)
    assert np.array_equal(scale.mean_,initial)
    model=RegisteredModel("BayesianRidge").fit(X[:72],y[:72])
    original=model.predict(X[72:])
    changed=X[72:].copy();changed[:,7]=1000000
    assert np.allclose(model.predict(changed),original)


@pytest.mark.parametrize("method",sorted(WEIGHTED))
def test_supported_sample_weights_reach_estimator(method):
    X,y=data()
    weight=np.ones(72);weight[:8]=5
    model=RegisteredModel(method).fit(X[:72],y[:72],sample_weight=weight)
    assert np.isfinite(model.predict(X[72:])).all()


def test_unsupported_weights_are_rejected_and_weighted_loss_changes_fit():
    X,y=data()
    for method in ("ARD","TheilSen","MultiTaskElasticNet","KNN","GaussianProcess"):
        with pytest.raises(WorkbenchError,match="权重"):
            RegisteredModel(method).fit(X[:72],y[:72],sample_weight=np.ones(72))
    y[0]+=20
    ordinary=RegisteredModel("BayesianRidge").fit(X[:72],y[:72])
    weight=np.ones(72);weight[0]=0
    weighted=RegisteredModel("BayesianRidge").fit(X[:72],y[:72],sample_weight=weight)
    assert not np.allclose(ordinary.predict(X[72:]),weighted.predict(X[72:]))


@pytest.mark.parametrize("method",("BayesianRidge","ARD","GaussianProcess","Quantile"))
def test_uncertainty_contract_preserves_units_and_does_not_claim_joint_coverage(method):
    X,y=data()
    model=RegisteredModel(method).fit(X[:72],y[:72])
    result=model.predict_uncertainty(X[72:])
    assert not result["calibrated"] and not result["joint_region"]
    if method=="Quantile":
        assert result["values"].shape==(24,3,2)
        assert np.allclose(result["values"][:,1,:],model.predict(X[72:]))
    else:
        assert result["std"].shape==(24,2) and np.all(result["std"]>=0)
        assert np.allclose(result["mean"],model.predict(X[72:]))
        scaled_X,scaled_y=X.copy(),y.copy();scaled_X[:,1]*=1000;scaled_y[:,1]*=1000
        scaled=RegisteredModel(method).fit(scaled_X[:72],scaled_y[:72]).predict_uncertainty(scaled_X[72:])
        assert np.allclose(scaled["mean"][:,1]/1000,result["mean"][:,1],rtol=1e-5,atol=1e-5)
        assert np.allclose(scaled["std"][:,1]/1000,result["std"][:,1],rtol=1e-4,atol=1e-5)


def test_new_uncertainty_is_saved_scored_and_returned_through_comparison(tmp_path,monkeypatch):
    from test_g2_models import SyntheticData,SyntheticService
    from copper_mvp.model_training import train_comparison
    from copper_mvp.model_evaluation import evaluate_comparison,read_training
    from copper_mvp.model_comparisons import ComparisonService
    from copper_mvp.model_registry import ComparisonPredictionRequest
    from copper_mvp.common import write_json,file_hash
    data_set=SyntheticData(tmp_path/"data")
    for module in ("model_training","model_evaluation","model_comparisons"):
        monkeypatch.setattr("copper_mvp."+module+".DataService",SyntheticService)
    root=tmp_path/("c"*32)
    request=ComparisonRequest(request_key="uncertainty",methods=("Persistence","BayesianRidge","Quantile"))
    train_comparison(data_set,request,root)
    evaluation=evaluate_comparison(data_set,root)
    assert len(evaluation["uncertainty_metrics"])==4
    assert all(not row["calibrated"] and not row["joint_region"] for row in evaluation["uncertainty_metrics"])
    write_json(root/"state.json",{"run_id":root.name,"status":"completed","created_at":"synthetic","fingerprint":"synthetic",
        "request":request.model_dump(mode="json"),"evaluation_sha256":file_hash(root/"evaluation.json")})
    event=next(iter(data_set.fold_for))
    answer=ComparisonService(tmp_path,data_set).predict(root.name,ComparisonPredictionRequest(event_id=event,method_id="BayesianRidge",include_uncertainty=True))
    assert answer["uncertainty"]["kind"]=="marginal_standard_deviation"
    with (root/"oof_uncertainty.csv").open("a") as stream:stream.write("\n")
    with pytest.raises(WorkbenchError,match="校验"):
        read_training(root)


def test_classical_study_api_checks_auth_and_bound_prediction_file(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from copper_mvp.api import create_app
    from copper_mvp.common import write_json,file_hash
    monkeypatch.delenv("COPPER_MOCK_URL",raising=False)
    monkeypatch.delenv("COPPER_MOCK_KEY_FILE",raising=False)
    monkeypatch.setenv("COPPER_ASSISTANT_LIVE_CALLS","0")
    with TestClient(create_app(run_dir=tmp_path/"api"),base_url="http://127.0.0.1") as client:
        wb=client.app.state.workbench
        identifier="d"*32
        root=wb.root/"classical_studies"/identifier
        root.mkdir(parents=True)
        write_json(root/"protocol.json",{"scope":"synthetic"})
        (root/"predictions.csv").write_bytes(b"event_id,value\nsynthetic,1\n")
        (root/"report.md").write_bytes(b"# Synthetic artifact\n")
        write_json(root/"evaluation.json",{"protocol_sha256":file_hash(root/"protocol.json"),"predictions_sha256":file_hash(root/"predictions.csv")})
        write_json(root/"state.json",{"id":identifier,"status":"completed","created_at":"2024-01-01T00:00:00+00:00",
            "evaluation_sha256":file_hash(root/"evaluation.json")})
        assert client.get("/api/v2/classical-studies").status_code==401
        key=wb.access.owner_key_path.read_text().strip()
        client.post("/api/auth/session",json={"access_code":key})
        assert client.get("/api/v2/classical-studies/"+identifier).status_code==200
        assert client.get("/api/v2/classical-studies/"+identifier+"/export").status_code==200
        (root/"predictions.csv").write_bytes(b"changed")
        assert client.get("/api/v2/classical-studies/"+identifier).status_code==409
