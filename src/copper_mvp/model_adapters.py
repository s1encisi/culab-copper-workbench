"""Six methods behind one two-target, physical-unit prediction interface."""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from copper_mvp.common import WorkbenchError
from copper_mvp.model_registry import FEATURE_COUNT, NUMERIC_COUNT, PRESETS, SEED, method_spec
from copper_mvp.modeling import make_model


def preprocess():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), list(range(NUMERIC_COUNT))),
        ("modes", OneHotEncoder(categories=[[0., 1., 2.]] * 4, handle_unknown="ignore", sparse_output=False),
         list(range(NUMERIC_COUNT, FEATURE_COUNT))),
    ])


def validate_X(values):
    X = np.asarray(values, dtype=float)
    if X.ndim != 2 or X.shape[1] != FEATURE_COUNT or not len(X):
        raise WorkbenchError("模型输入必须为非空的 114 字段矩阵", "MODEL_INPUT_SHAPE")
    if np.isinf(X).any() or not np.isfinite(X[:, :2]).all() or not np.isin(X[:, NUMERIC_COUNT:], [0., 1., 2.]).all():
        raise WorkbenchError("当前浓度或工况编码无效，或输入包含无穷值", "MODEL_INPUT_VALUES")
    return X


class RegisteredModel:
    def __init__(self, method_id: str, seed: int = SEED):
        self.method_id = method_id
        self.seed = seed
        self.spec = method_spec(method_id, seed)
        self.models = []
        self.fit_warnings = []
        self.is_fitted = method_id == "Persistence"
        self.n_features_in_ = FEATURE_COUNT

    def fit(self, X, y):
        X = validate_X(X)
        if not self.spec["requires_fit"]:
            return self
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X), 2) or not np.isfinite(y).all():
            raise WorkbenchError("训练目标必须是有限的 Cu/As 双列矩阵", "MODEL_TARGET_VALUES")
        delta = y - X[:, :2]
        self.models = []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if self.method_id == "PLS":
                model = TransformedTargetRegressor(
                    regressor=Pipeline([("preprocess", preprocess()), ("regressor", PLSRegression(**PRESETS["PLS"]))]),
                    transformer=StandardScaler())
                model.fit(X, delta)
                self.models = [model]
            else:
                for target in range(2):
                    if self.method_id in ("DeltaRidge", "DeltaHGB"):
                        model = make_model(self.method_id)
                        if self.method_id == "DeltaHGB":
                            model.set_params(random_state=self.seed)
                    else:
                        estimator = ElasticNet(**PRESETS["ElasticNet"]) if self.method_id == "ElasticNet" else HuberRegressor(**PRESETS["Huber"])
                        model = TransformedTargetRegressor(
                            regressor=Pipeline([("preprocess", preprocess()), ("regressor", estimator)]),
                            transformer=StandardScaler())
                    model.fit(X, delta[:, target])
                    self.models.append(model)
        self.fit_warnings = [{"category": w.category.__name__, "message": str(w.message)[:400]} for w in captured]
        self.is_fitted = True
        return self

    def predict(self, X):
        X = validate_X(X)
        if not self.is_fitted:
            raise WorkbenchError("模型尚未拟合", "MODEL_NOT_FITTED")
        if self.method_id == "Persistence":
            result = X[:, :2].copy()
        elif self.method_id == "PLS":
            result = X[:, :2] + np.asarray(self.models[0].predict(X)).reshape(len(X), 2)
        else:
            result = X[:, :2] + np.column_stack([model.predict(X).reshape(-1) for model in self.models])
        if result.shape != (len(X), 2) or not np.isfinite(result).all():
            raise WorkbenchError("模型输出不是有限的 Cu/As 双列结果", "MODEL_OUTPUT_VALUES")
        return result
