from __future__ import annotations

import joblib
import numpy as np
import pytest
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_registry import LEGACY_METHOD_IDS, ComparisonRequest


@pytest.mark.parametrize("name", ["CatBoost", "NGBoost", "EBM"])
def test_specialized_registered_model_roundtrip_and_probability(tmp_path, name, monkeypatch):
    from copper_mvp.specialized_models import specialized_dependencies
    specialized_dependencies()
    if name == "NGBoost":
        import ngboost.ngboost as module
        def forbidden_split(*args, **kwargs):
            raise AssertionError("Unexpected random validation split")
        monkeypatch.setattr(module, "train_test_split", forbidden_split)
    rng = np.random.default_rng(29)
    X = rng.normal(size=(80,114));X[:,0]=4+0.2*X[:,0];X[:,1]=300+10*X[:,1]
    X[:,110:]=rng.integers(0,3,(80,4));X[::7,5]=np.nan
    y = X[:,:2]+np.column_stack((0.3*X[:,2],20*X[:,3]))
    model = RegisteredModel(name,29).fit(X,y,sample_weight=np.linspace(0.5,1.5,len(X)))
    values = model.predict(X[:12]);assert values.shape == (12,2) and np.isfinite(values).all()
    np.testing.assert_allclose(model.fit_metadata["target_delta_mean"], (y-X[:,:2]).mean(axis=0))
    path = tmp_path/"model.joblib";joblib.dump(model,path);restored=joblib.load(path)
    np.testing.assert_allclose(restored.predict(X[:12]),values)
    if name == "CatBoost":
        assert model.fit_metadata["native_has_time"] == [True,True]
        assert model.fit_metadata["native_boosting_type"] == ["Ordered","Ordered"]
    if name == "NGBoost":
        uncertainty = restored.predict_uncertainty(X[:12])
        np.testing.assert_allclose(uncertainty["mean"],values)
        assert (uncertainty["std"]>0).all() and not uncertainty["calibrated"]
    if name == "EBM":
        assert all(n>=114 for n in model.fit_metadata["term_counts"])
    assert ComparisonRequest(request_key="default").methods == LEGACY_METHOD_IDS
