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

from copper_mvp.bart_registry import BART_METHODS
from copper_mvp.classical_registry import CLASSICAL_METHODS
from copper_mvp.common import WorkbenchError
from copper_mvp.model_registry import FEATURE_COUNT, NUMERIC_COUNT, PRESETS, SEED, method_spec
from copper_mvp.modeling import make_model
from copper_mvp.specialized_registry import SPECIALIZED_METHODS
from copper_mvp.statistical_registry import SEQUENCE_METHODS, STATISTICAL_METHODS
from copper_mvp.symbolic_registry import SYMBOLIC_METHODS
from copper_mvp.tabpfn_registry import TABPFN_METHODS
from copper_mvp.tabular_registry import TABULAR_METHODS
from copper_mvp.temporal_registry import TEMPORAL_METHODS


def preprocess():
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(range(NUMERIC_COUNT)),
            ),
            (
                "modes",
                OneHotEncoder(categories=[[0.0, 1.0, 2.0]] * 4, handle_unknown="ignore", sparse_output=False),
                list(range(NUMERIC_COUNT, FEATURE_COUNT)),
            ),
        ]
    )


def validate_X(values):
    X = np.asarray(values, dtype=float)
    if X.ndim != 2 or X.shape[1] != FEATURE_COUNT or not len(X):
        raise WorkbenchError("模型输入必须为非空的 114 字段矩阵", "MODEL_INPUT_SHAPE")
    if np.isinf(X).any() or not np.isfinite(X[:, :2]).all() or not np.isin(X[:, NUMERIC_COUNT:], [0.0, 1.0, 2.0]).all():
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

    def fit(self, X, y, sample_weight=None, *, max_wall_seconds=None):
        if self.method_id in SEQUENCE_METHODS + TEMPORAL_METHODS:
            raise WorkbenchError("序列方法需要完整事件上下文", "MODEL_CONTEXT_REQUIRED")
        X = validate_X(X)
        weight = None
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=float)
            if weight.shape != (len(X),) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
                raise WorkbenchError("训练权重必须有限、非负且总和为正", "MODEL_SAMPLE_WEIGHT")
            if self.method_id in ("Persistence", "PLS") or self.method_id in STATISTICAL_METHODS:
                raise WorkbenchError("该方法不支持训练样本权重", "MODEL_SAMPLE_WEIGHT_UNSUPPORTED")
        if not self.spec["requires_fit"]:
            return self
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X), 2) or not np.isfinite(y).all():
            raise WorkbenchError("训练目标必须是有限的 Cu/As 双列矩阵", "MODEL_TARGET_VALUES")
        if self.method_id in TABPFN_METHODS:
            from copper_mvp.tabpfn_models import TabPFNModel

            self._tabpfn = TabPFNModel(self.seed).fit(X, y, weight, max_wall_seconds=max_wall_seconds)
            self.models, self.fit_warnings = self._tabpfn.models, self._tabpfn.fit_warnings
            self.fit_metadata = self._tabpfn.fit_metadata
            self.is_fitted = True
            return self
        if self.method_id in TABULAR_METHODS:
            from copper_mvp.tabular_models import TabularModel

            self._tabular = TabularModel(self.method_id, self.seed).fit(X, y, weight, max_wall_seconds=max_wall_seconds)
            self.models, self.fit_warnings = self._tabular.models, self._tabular.fit_warnings
            self.fit_metadata = self._tabular.fit_metadata
            self.is_fitted = True
            return self
        if self.method_id in SYMBOLIC_METHODS:
            from copper_mvp.symbolic_models import SymbolicModel

            self._symbolic = SymbolicModel(self.seed).fit(X, y, weight, max_wall_seconds=max_wall_seconds)
            self.models, self.fit_warnings = self._symbolic.models, self._symbolic.fit_warnings
            self.fit_metadata = self._symbolic.fit_metadata
            self.is_fitted = True
            return self
        if self.method_id in BART_METHODS:
            from copper_mvp.bart_models import BartModel

            self._bart = BartModel(self.seed).fit(X, y, weight, max_wall_seconds=max_wall_seconds)
            self.models, self.fit_warnings = self._bart.models, self._bart.fit_warnings
            self.fit_metadata = self._bart.fit_metadata
            self.is_fitted = True
            return self
        if self.method_id in STATISTICAL_METHODS:
            from copper_mvp.statistical_models import StatisticalRegressor

            self._statistical = StatisticalRegressor(self.method_id, self.seed).fit(X, y)
            self.models = self._statistical.models
            self.fit_warnings = self._statistical.fit_warnings
            self.fit_metadata = self._statistical.fit_metadata
            self.is_fitted = True
            return self
        if self.method_id in SPECIALIZED_METHODS:
            from copper_mvp.specialized_models import SpecializedModel

            self._specialized = SpecializedModel(self.method_id, self.seed).fit(X, y, weight)
            self.models = self._specialized.models
            self.fit_warnings = self._specialized.fit_warnings
            self.fit_metadata = self._specialized.fit_metadata
            self.is_fitted = True
            return self
        if self.method_id in CLASSICAL_METHODS:
            from copper_mvp.classical_models import ClassicalModel

            self._classical = ClassicalModel(self.method_id, self.seed).fit(X, y, weight)
            self.models = self._classical.models
            self.fit_warnings = self._classical.fit_warnings
            self.is_fitted = True
            return self
        delta = y - X[:, :2]
        self.models = []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if self.method_id == "PLS":
                model = TransformedTargetRegressor(
                    regressor=Pipeline([("preprocess", preprocess()), ("regressor", PLSRegression(**PRESETS["PLS"]))]),
                    transformer=StandardScaler(),
                )
                model.fit(X, delta)
                self.models = [model]
            else:
                for target in range(2):
                    if self.method_id in ("DeltaRidge", "DeltaHGB"):
                        model = make_model(self.method_id)
                        if self.method_id == "DeltaHGB":
                            model.set_params(random_state=self.seed)
                    else:
                        estimator = (
                            ElasticNet(**PRESETS["ElasticNet"])
                            if self.method_id == "ElasticNet"
                            else HuberRegressor(**PRESETS["Huber"])
                        )
                        model = TransformedTargetRegressor(
                            regressor=Pipeline([("preprocess", preprocess()), ("regressor", estimator)]),
                            transformer=StandardScaler(),
                        )
                    kwargs = (
                        {}
                        if weight is None
                        else {"sample_weight" if self.method_id == "DeltaHGB" else "regressor__sample_weight": weight}
                    )
                    model.fit(X, delta[:, target], **kwargs)
                    self.models.append(model)
        self.fit_warnings = [{"category": w.category.__name__, "message": str(w.message)[:400]} for w in captured]
        self.is_fitted = True
        return self

    def predict(self, X):
        if self.method_id in SEQUENCE_METHODS + TEMPORAL_METHODS:
            raise WorkbenchError("序列方法需要事件标识和原始记录顺序", "MODEL_CONTEXT_REQUIRED")
        X = validate_X(X)
        if not self.is_fitted:
            raise WorkbenchError("模型尚未拟合", "MODEL_NOT_FITTED")
        if self.method_id in TABPFN_METHODS:
            result = self._tabpfn.predict(X)
        elif self.method_id in TABULAR_METHODS:
            result = self._tabular.predict(X)
        elif self.method_id in SYMBOLIC_METHODS:
            result = self._symbolic.predict(X)
        elif self.method_id in BART_METHODS:
            result = self._bart.predict(X)
        elif self.method_id in SPECIALIZED_METHODS:
            result = self._specialized.predict(X)
        elif self.method_id in STATISTICAL_METHODS:
            result = self._statistical.predict(X)
        elif self.method_id in CLASSICAL_METHODS:
            result = self._classical.predict(X)
        elif self.method_id == "Persistence":
            result = X[:, :2].copy()
        elif self.method_id == "PLS":
            result = X[:, :2] + np.asarray(self.models[0].predict(X)).reshape(len(X), 2)
        else:
            result = X[:, :2] + np.column_stack([model.predict(X).reshape(-1) for model in self.models])
        if result.shape != (len(X), 2) or not np.isfinite(result).all():
            raise WorkbenchError("模型输出不是有限的 Cu/As 双列结果", "MODEL_OUTPUT_VALUES")
        return result

    def predict_uncertainty(self, X):
        X = validate_X(X)
        if not self.is_fitted:
            raise WorkbenchError("模型尚未拟合", "MODEL_NOT_FITTED")
        if self.method_id in TABPFN_METHODS:
            return self._tabpfn.uncertainty(X)
        if self.method_id in BART_METHODS:
            return self._bart.uncertainty(X)
        if self.method_id in SPECIALIZED_METHODS:
            return self._specialized.uncertainty(X)
        if self.method_id not in CLASSICAL_METHODS:
            raise WorkbenchError("该方法未开放概率输出", "MODEL_UNCERTAINTY_UNSUPPORTED")
        return self._classical.uncertainty(X)

    def fit_context(self, data, train_ids, cutoff, *, max_wall_seconds=None):
        if self.method_id in TEMPORAL_METHODS:
            from copper_mvp.temporal_models import TemporalModel

            self._temporal = TemporalModel(self.method_id, self.seed).fit_context(
                data, train_ids, cutoff, max_wall_seconds=max_wall_seconds
            )
            self.models, self.fit_warnings = self._temporal.models, self._temporal.fit_warnings
            self.fit_metadata = self._temporal.fit_metadata
            self.is_fitted = True
            return self
        if self.method_id not in SEQUENCE_METHODS:
            raise WorkbenchError("该方法使用矩阵训练接口", "MODEL_CONTEXT_UNSUPPORTED")
        from copper_mvp.statistical_models import EventSequenceModel

        self._sequence = EventSequenceModel(self.method_id, self.seed).fit_context(data, train_ids, cutoff)
        self.models = self._sequence.models
        self.fit_warnings = self._sequence.fit_warnings
        self.fit_metadata = self._sequence.fit_metadata
        self.is_fitted = True
        return self

    def predict_context(self, data, event_ids):
        if not self.is_fitted:
            raise WorkbenchError("模型尚未拟合", "MODEL_NOT_FITTED")
        if self.method_id in TEMPORAL_METHODS:
            return self._temporal.predict_context(data, event_ids)
        if self.method_id in SEQUENCE_METHODS:
            return self._sequence.forecast_context(data, event_ids)["mean"]
        return self.predict(data.X.loc[event_ids].to_numpy(float))

    def predict_uncertainty_context(self, data, event_ids):
        if not self.is_fitted:
            raise WorkbenchError("模型尚未拟合", "MODEL_NOT_FITTED")
        if self.method_id in TEMPORAL_METHODS:
            raise WorkbenchError("该时序方法尚未提供校准区间", "MODEL_UNCERTAINTY_UNSUPPORTED")
        if self.method_id in SEQUENCE_METHODS:
            return self._sequence.forecast_context(data, event_ids)
        return self.predict_uncertainty(data.X.loc[event_ids].to_numpy(float))
