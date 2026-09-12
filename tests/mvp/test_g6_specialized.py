from __future__ import annotations

import joblib
import numpy as np
import pytest
from copper_mvp.model_adapters import RegisteredModel
from copper_mvp.model_registry import LEGACY_METHOD_IDS, ComparisonRequest


@pytest.mark.parametrize("name", ["CatBoost", "NGBoost", "EBM", "Cubist"])
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
    weight = None if name == "Cubist" else np.linspace(0.5,1.5,len(X))
    model = RegisteredModel(name,29).fit(X,y,sample_weight=weight)
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
        from copper_mvp.specialized_models import native_matrix
        for native in model.models:
            additive = native.eval_terms(native_matrix(X[:12])).sum(axis=1)+native.intercept_
            np.testing.assert_allclose(additive, native.predict(native_matrix(X[:12])))
    if name == "Cubist":
        import re
        import pandas as pd
        import cubist.cubist as native_module
        from copper_mvp.common import WorkbenchError
        raw = model.models[0].model_
        broken, count = re.subn(r'\bredn=(?:"[^"]*"|\S+)', 'redn="-nan(ind)"', raw, count=1)
        assert count == 1
        parsed = native_module._parse_model(broken, model.models[0].feature_names_in_)
        original = native_module._copper_original_parser(broken.replace('-nan(ind)', 'nan'), model.models[0].feature_names_in_)
        for index in (1, 2, 3):
            pd.testing.assert_frame_equal(parsed[index], original[index])
        assert np.isnan(parsed[4]) and model.models[0].model_ == raw
        assert model._specialized._cubist.rules()["models"] == [m.model_ for m in model.models]
        with pytest.raises(WorkbenchError, match="权重"):
            RegisteredModel(name,29).fit(X,y,sample_weight=np.ones(len(X)))
    assert ComparisonRequest(request_key="default").methods == LEGACY_METHOD_IDS
