"""Cubist rule/linear models in physical two-target units."""

import os
import re
import sys
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from copper_mvp.classical_models import classical_preprocess
from copper_mvp.common import PROJECT_ROOT, WorkbenchError


def cubist_dependency():
    target = Path(os.environ.get("COPPER_CUBIST_RUNTIME_DIR", str(PROJECT_ROOT / "runs/dependencies/cubist-1.2.2")))
    if str(target) not in sys.path and target.is_dir():
        sys.path.insert(0, str(target))
    try:
        version = metadata.version("cubist")
    except metadata.PackageNotFoundError:
        raise WorkbenchError("Cubist 1.2.2 可选依赖尚未安装", "CUBIST_DEPENDENCY") from None
    if version != "1.2.2":
        raise WorkbenchError("Cubist 版本与协议不同", "CUBIST_DEPENDENCY")
    import cubist.cubist as native_module

    if not hasattr(native_module, "_copper_original_parser"):
        original = native_module._parse_model
        native_module._copper_original_parser = original

        def parse_with_windows_diagnostic(model, feature_names):
            # Only the undefined committee reduction statistic is normalized.
            # The original model text and prediction coefficients remain intact.
            metadata_text = re.sub(r'(\bredn="?)[+-]?nan\(ind\)("?)', r"\1nan\2", model, flags=re.IGNORECASE)
            return original(model=metadata_text, feature_names=feature_names)

        native_module._parse_model = parse_with_windows_diagnostic


class CubistModel:
    def __init__(self, seed):
        self.seed = seed
        self.models = []
        self.fit_warnings = []

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise WorkbenchError("Cubist 首版未开放样本权重", "MODEL_SAMPLE_WEIGHT_UNSUPPORTED")
        self.models = []
        cubist_dependency()
        from cubist import Cubist

        from copper_mvp.specialized_registry import PARAMETERS

        X, y = np.asarray(X, float), np.asarray(y, float)
        self.preprocessor = classical_preprocess("Cubist")
        inputs = np.ascontiguousarray(self.preprocessor.fit_transform(X))
        self.target_scaler = StandardScaler().fit(y - X[:, :2])
        target = self.target_scaler.transform(y - X[:, :2])
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            for column in range(2):
                model = Cubist(**PARAMETERS["Cubist"], random_state=self.seed)
                model.fit(inputs, np.ascontiguousarray(target[:, column]))
                self.models.append(model)
        self.fit_warnings = [{"category": w.category.__name__, "message": str(w.message)[:400]} for w in notices]
        self.fit_metadata = {
            "training_rows": len(X),
            "preprocessed_columns": inputs.shape[1],
            "target_delta_mean": self.target_scaler.mean_.tolist(),
            "target_delta_scale": self.target_scaler.scale_.tolist(),
            "rule_linear_models": True,
            "internal_cv": False,
            "sample_weight": "unsupported",
            "undefined_committee_reduction": [
                not np.isfinite(m.committee_error_reduction_) if m.committee_error_reduction_ is not None else True
                for m in self.models
            ],
        }
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        inputs = np.ascontiguousarray(self.preprocessor.transform(X))
        prediction = np.column_stack(
            [model.predict(pd.DataFrame(inputs, columns=model.feature_names_in_)) for model in self.models]
        )
        return X[:, :2] + self.target_scaler.inverse_transform(prediction)

    def rules(self):
        return {
            "feature_names": self.preprocessor.get_feature_names_out().tolist(),
            "target_delta_mean": self.target_scaler.mean_.tolist(),
            "target_delta_scale": self.target_scaler.scale_.tolist(),
            "models": [model.model_ for model in self.models],
            "scope": "rules in transformed feature space; model associations",
        }
