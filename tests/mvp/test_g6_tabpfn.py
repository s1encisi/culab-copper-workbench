import numpy as np
import pytest
import joblib
import sklearn
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_registry import catalog
from copper_mvp.tabpfn_client import close_worker
from copper_mvp.common import WorkbenchError


def test_final_method_is_registered_without_counting_context_variants():
    data=catalog()
    assert len({r["method_id"] for r in data["items"]})==38
    assert data["new_method_count"]==35
    assert next(r for r in data["items"] if r["method_id"]=="TabPFN")["attribution"]=="Built with PriorLabs-TabPFN"


def test_tabpfn_rejects_unverified_weights():
    with pytest.raises(WorkbenchError,match="权重"):
        RegisteredModel("TabPFN",17).fit(np.zeros((20,114)),np.zeros((20,2)),sample_weight=np.ones(20))


def test_isolated_context_cache_and_roundtrip(tmp_path):
    rng=np.random.default_rng(23)
    X=rng.normal(size=(48,114));X[:,0]=4+.2*X[:,0];X[:,1]=300+10*X[:,1]
    X[:,110:]=rng.integers(0,3,(48,4))
    y=X[:,:2]+np.column_stack((.2*X[:,2],20*X[:,3]))
    try:
        first=RegisteredModel("TabPFN",23).fit(X,y)
        reference=first.predict(X[:4])
        second=RegisteredModel("TabPFN",24).fit(X,y+.1)
        assert np.isfinite(second.predict(X[:4])).all()
        np.testing.assert_allclose(first.predict(X[:4]),reference,rtol=1e-6,atol=1e-6)
        path=tmp_path/"context.joblib";joblib.dump(first,path)
        restored=joblib.load(path)
        np.testing.assert_allclose(restored.predict(X[:4]),reference,rtol=1e-6,atol=1e-6)
        q=restored.predict_uncertainty(X[:4])["values"]
        assert q.shape==(4,3,2) and (np.diff(q,axis=1)>=0).all()
        assert sklearn.__version__=="1.7.2"
    finally:
        close_worker()
