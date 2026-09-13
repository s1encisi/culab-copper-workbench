"""Native CatBoost, NGBoost, EBM and Cubist in the physical-unit interface."""
from importlib import metadata
import sys
import os
from pathlib import Path
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from copper_mvp.common import PROJECT_ROOT, WorkbenchError
from copper_mvp.classical_models import classical_preprocess
from copper_mvp.specialized_registry import PACKAGES, PARAMETERS


def specialized_dependencies():
    target = Path(os.environ.get("COPPER_SPECIALIZED_RUNTIME_DIR", str(PROJECT_ROOT / "runs/dependencies/specialized-models-v1")))
    if target.is_dir() and str(target) not in sys.path:
        sys.path.insert(0, str(target))
    from copper_mvp.cubist_model import cubist_dependency
    cubist_dependency()
    for name, expected in PACKAGES.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise WorkbenchError("请运行 scripts/setup_specialized_models.ps1", "SPECIALIZED_DEPENDENCY") from None
        if actual != expected:
            raise WorkbenchError("专用模型依赖版本不一致: " + name, "SPECIALIZED_DEPENDENCY")


def native_matrix(X):
    result = np.asarray(X, dtype=object).copy()
    for column in range(110, 114):
        result[:, column] = [str(int(value)) for value in result[:, column]]
    return result


class SpecializedModel:
    def __init__(self, method, seed):
        self.method_id, self.seed = method, seed
        self.models = []
        self.fit_warnings = []

    def fit(self, X, y, sample_weight=None):
        specialized_dependencies()
        if self.method_id == "Cubist":
            from copper_mvp.cubist_model import CubistModel
            self._cubist = CubistModel(self.seed).fit(X, y, sample_weight)
            self.models = self._cubist.models
            self.fit_warnings = self._cubist.fit_warnings
            self.fit_metadata = self._cubist.fit_metadata
            return self
        X, y = np.asarray(X, float), np.asarray(y, float)
        sample_weight = None if sample_weight is None else np.ascontiguousarray(sample_weight, dtype=float)
        self.target_scaler = StandardScaler().fit(y-X[:, :2])
        target = self.target_scaler.transform(y-X[:, :2])
        if self.method_id == "NGBoost":
            self.preprocessor = classical_preprocess("NGBoost")
            inputs = self.preprocessor.fit_transform(X)
        else:
            inputs = native_matrix(X)
        params = dict(PARAMETERS[self.method_id])
        self.models = []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for column in range(2):
                target_column = np.ascontiguousarray(target[:, column])
                if self.method_id == "CatBoost":
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(**params, random_seed=self.seed)
                    model.fit(inputs, target_column, cat_features=list(range(110,114)), sample_weight=sample_weight)
                elif self.method_id == "NGBoost":
                    from ngboost import NGBRegressor
                    from ngboost.distns import Normal
                    from ngboost.scores import LogScore
                    model = NGBRegressor(**params, Dist=Normal, Score=LogScore,
                        Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=self.seed), random_state=self.seed)
                    model.fit(inputs, target_column, sample_weight=sample_weight)
                elif self.method_id == "EBM":
                    from interpret.glassbox import ExplainableBoostingRegressor
                    model = ExplainableBoostingRegressor(**params, random_state=self.seed,
                        feature_types=["continuous"]*110+["nominal"]*4)
                    model.fit(inputs, target_column, sample_weight=sample_weight)
                else:
                    raise WorkbenchError("没有该专用模型", "METHOD_NOT_FOUND")
                self.models.append(model)
        self.fit_warnings = [{"category":w.category.__name__,"message":str(w.message)[:400]} for w in captured]
        self.fit_metadata = {"target_delta_mean": self.target_scaler.mean_.tolist(),
            "target_delta_scale": self.target_scaler.scale_.tolist(), "training_rows": len(X),
            "input_order_used": self.method_id == "CatBoost", "internal_validation": "disabled"}
        if self.method_id == "CatBoost":
            self.fit_metadata["native_has_time"] = [m.get_all_params()["has_time"] for m in self.models]
            self.fit_metadata["native_boosting_type"] = [m.get_all_params()["boosting_type"] for m in self.models]
        if self.method_id == "EBM":
            self.fit_metadata["term_counts"] = [len(m.term_features_) for m in self.models]
            self.fit_metadata["interaction_counts"] = [sum(len(t)>1 for t in m.term_features_) for m in self.models]
        return self

    def _inputs(self, X):
        return self.preprocessor.transform(X) if self.method_id == "NGBoost" else native_matrix(X)

    def predict(self, X):
        if self.method_id == "Cubist":
            return self._cubist.predict(X)
        X = np.asarray(X, float)
        inputs = self._inputs(X)
        prediction = np.column_stack([m.predict(inputs) for m in self.models])
        return X[:, :2]+self.target_scaler.inverse_transform(prediction)

    def uncertainty(self, X):
        if self.method_id != "NGBoost":
            raise WorkbenchError("该方法未声明概率输出", "MODEL_UNCERTAINTY_UNSUPPORTED")
        X = np.asarray(X, float);inputs = self._inputs(X)
        distributions = [m.pred_dist(inputs) for m in self.models]
        mean = X[:, :2]+self.target_scaler.inverse_transform(np.column_stack([d.loc for d in distributions]))
        std = np.column_stack([d.scale for d in distributions])*self.target_scaler.scale_
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise WorkbenchError("NGBoost 分布参数无效", "MODEL_UNCERTAINTY_VALUES")
        return {"kind":"marginal_standard_deviation", "distribution":"normal", "mean":mean, "std":std,
                "calibrated":False, "joint_region":False}
