"""Statistical regressors and causal event-step state forecasts."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer, StandardScaler

from copper_mvp.classical_models import StableNumericScale
from copper_mvp.common import WorkbenchError, digest
from copper_mvp.data_contracts import source_time
from copper_mvp.statistical_runtime import statistical_dependencies


class StatisticalRegressor:
    def __init__(self, method, seed):
        self.method_id, self.seed = method, seed
        self.fit_warnings = []

    def fit(self, X, y):
        statistical_dependencies()
        if len(X) < 30:
            raise WorkbenchError("统计回归训练样本不足", "MODEL_TRAINING_SUPPORT")
        self.xscale = StableNumericScale().fit(np.asarray(X)[:, :2])
        A = self.xscale.transform(np.asarray(X)[:, :2])
        self.yscale = StandardScaler().fit(np.asarray(y) - np.asarray(X)[:, :2])
        target = self.yscale.transform(np.asarray(y) - np.asarray(X)[:, :2])
        self.models = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if self.method_id == "LocalLinearKernel":
                from statsmodels.nonparametric.kernel_regression import KernelReg

                self.models = [KernelReg(target[:, t], A, var_type="cc", reg_type="ll", bw=[0.5, 0.5]) for t in (0, 1)]
            else:
                from statsmodels.regression.linear_model import OLS

                self.splines = SplineTransformer(
                    n_knots=5, degree=3, knots="quantile", include_bias=False, extrapolation="linear"
                )
                basis = self.splines.fit_transform(A)
                width = basis.shape[1] // 2
                block = np.diff(np.eye(width), n=2, axis=0)
                penalty = np.zeros((2 * len(block), 1 + basis.shape[1]))
                penalty[: len(block), 1 : 1 + width] = block
                penalty[len(block) :, 1 + width :] = block
                augmented = np.vstack((np.column_stack((np.ones(len(A)), basis)), penalty))
                self.models = [OLS(np.r_[target[:, t], np.zeros(len(penalty))], augmented).fit() for t in (0, 1)]
        self.fit_warnings = [{"category": w.category.__name__, "message": str(w.message)[:400]} for w in caught]
        self.fit_metadata = {
            "input_columns": [0, 1],
            "preprocessing": "training_only_current_result_pair",
            "penalty_rows_are_observations": False,
            "training_events": len(X),
        }
        return self

    def predict(self, X):
        A = self.xscale.transform(np.asarray(X)[:, :2])
        if self.method_id == "LocalLinearKernel":
            delta = np.column_stack([m.fit(A)[0] for m in self.models])
        else:
            basis = np.column_stack((np.ones(len(A)), self.splines.transform(A)))
            delta = np.column_stack([m.predict(basis) for m in self.models])
        return np.asarray(X)[:, :2] + self.yscale.inverse_transform(delta)


def event_arrays(data, with_exog):
    ids = list(data.frame.index)
    times = [source_time(value) for value in data.frame.loc[ids, "decision_at"]]
    if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
        raise WorkbenchError("原始事件索引不是时间顺序", "SEQUENCE_ORDER")
    Y = data.X.loc[ids].iloc[:, :2].to_numpy(float, copy=True)
    quality = (
        data.frame.loc[ids, "origin_result_quality_eligible"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    )
    Y[~quality] = np.nan
    Y[~np.isfinite(Y) | (Y < 0)] = np.nan
    exog = None
    if with_exog:
        names = ("stage3_current_a__t_minus_0h", "stage4_current_a__t_minus_0h")
        exog = data.X.loc[ids, list(names)].to_numpy(float, copy=True)
        for i, event in enumerate(ids):
            admission = data.admissions[event]
            if admission["admission_status"] == "REJECTED" or source_time(admission["feature_cutoff_at"]) > times[i]:
                exog[i] = np.nan
        ages = data.frame.loc[ids, [name + "__age_h" for name in names]].to_numpy(float)
        exog[~np.isfinite(ages) | (ages < 0)] = np.nan
    return ids, times, Y, exog


def forward_fill(values, initial):
    result = np.asarray(values, float).copy()
    last = np.asarray(initial, float).copy()
    for i, row in enumerate(result):
        good = np.isfinite(row)
        last[good] = row[good]
        result[i] = last
    return result


class EventSequenceModel:
    def __init__(self, method, seed):
        self.method_id, self.seed = method, seed
        self.fit_warnings = []
        self._last_prediction = None

    def _prefix_hash(self, ids, times, Y, exog, end):
        return digest(
            {
                "ids": ids[:end],
                "times": [t.isoformat() for t in times[:end]],
                "Y": Y[:end].tolist(),
                "exog": exog[:end].tolist() if exog is not None else None,
            }
        )

    def fit_context(self, data, train_ids, cutoff):
        statistical_dependencies()
        cutoff = source_time(cutoff)
        ids, times, Y, exog = event_arrays(data, self.method_id == "SARIMAX")
        count = sum(t <= cutoff for t in times)
        if count < 30 or np.any(np.isfinite(Y[:count]).sum(axis=0) < 30):
            raise WorkbenchError("序列训练缺少足够已知化验", "MODEL_TRAINING_SUPPORT")
        self.fit_cutoff_at = cutoff.isoformat()
        self.train_count = count
        self.last_training_index = count - 1
        self.training_event_ids = ids[:count]
        self.feature_spec_hash = digest(data.feature_columns)
        self.training_prefix_hash = self._prefix_hash(ids, times, Y, exog, count)
        self.mean_ = np.nanmean(Y[:count], axis=0)
        self.scale_ = np.nanstd(Y[:count], axis=0)
        self.scale_ = np.where(self.scale_ > 0, self.scale_, 1.0)
        normalized = (Y[:count] - self.mean_) / self.scale_
        self.models = []
        diagnostics = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if self.method_id == "VAR":
                from statsmodels.tsa.api import VAR

                filled = forward_fill(normalized, np.zeros(2))
                result = VAR(filled).fit(maxlags=2, trend="c")
                self.models = [result]
                self.last_observed = filled[-1].copy()
                self.var_history = filled[-result.k_ar :].copy()
                diagnostics.append(
                    {
                        "aic": float(result.aic),
                        "bic": float(result.bic),
                        "stable": bool(result.is_stable()),
                        "lags": result.k_ar,
                    }
                )
            elif self.method_id == "SARIMAX":
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                lagged = np.vstack((np.full((1, 2), np.nan), exog[: count - 1]))
                self.exog_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                filled = self.exog_imputer.fit_transform(lagged)
                self.exog_scale = StableNumericScale().fit(filled)
                regressors = np.column_stack((np.ones(len(filled)), self.exog_scale.transform(filled)))
                for target in (0, 1):
                    result = SARIMAX(
                        normalized[:, target],
                        exog=regressors,
                        order=(1, 0, 1),
                        trend="n",
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    ).fit(disp=False, maxiter=200)
                    self.models.append(result)
                    diagnostics.append(
                        {
                            "target": target,
                            "aic": float(result.aic),
                            "bic": float(result.bic),
                            "converged": bool(result.mle_retvals.get("converged", False)),
                        }
                    )
            else:
                from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

                for target in (0, 1):
                    model = ExponentialSmoothing(
                        normalized[:, target],
                        trend=True,
                        damped_trend=True,
                        seasonal=None,
                        initialization_method="estimated",
                        concentrate_scale=False,
                    )
                    initial_values = forward_fill(normalized[:, target : target + 1], np.zeros(1))[:, 0]
                    initializer = ExponentialSmoothing(
                        initial_values,
                        trend=True,
                        damped_trend=True,
                        seasonal=None,
                        initialization_method="estimated",
                        concentrate_scale=False,
                    )
                    start_params = initializer.start_params
                    start_params[model.param_names.index("sigma2")] = max(float(np.nanvar(normalized[:, target])), 1e-8)
                    result = model.fit(start_params=start_params, disp=False, maxiter=200)
                    self.models.append(result)
                    diagnostics.append(
                        {
                            "target": target,
                            "aic": float(result.aic),
                            "bic": float(result.bic),
                            "converged": bool(result.mle_retvals.get("converged", False)),
                        }
                    )
        self.fit_warnings = [{"category": w.category.__name__, "message": str(w.message)[:400]} for w in caught]
        self.fit_metadata = {
            "fit_cutoff_at": self.fit_cutoff_at,
            "training_event_ids_hash": digest(self.training_event_ids),
            "training_prefix_hash": self.training_prefix_hash,
            "raw_series_records": count,
            "supervised_primary_event_count": len(train_ids),
            "supervised_primary_event_hash": digest(train_ids),
            "observed_per_target": np.isfinite(Y[:count]).sum(axis=0).tolist(),
            "missing_per_target": np.isnan(Y[:count]).sum(axis=0).tolist(),
            "original_event_index_preserved": True,
            "step_basis": "recorded_events_not_fixed_hours",
            "parameter_training_scope": "all quality-eligible current assay observations available at the fit cutoff",
            "missing_policy": "causal_forward_fill_with_training_mean_initialization"
            if self.method_id == "VAR"
            else "missing_observation_state_filter",
            "diagnostics": diagnostics,
            "parameter_updates_during_prediction": False,
            "prediction_state_method": "fixed_parameter_forward_filter_predicted_information"
            if self.method_id != "VAR"
            else "fixed_VAR_coefficients_causal_lag_update",
            "ETS_initialization": "causal fill for optimizer starting states; likelihood retains missing observations"
            if self.method_id == "ETS"
            else None,
        }
        return self

    def forecast_context(self, data, events):
        ids, times, Y, exog = event_arrays(data, self.method_id == "SARIMAX")
        positions = {e: i for i, e in enumerate(ids)}
        query = [positions[e] for e in events]
        if min(query) < self.last_training_index:
            raise WorkbenchError("模型参数使用了该事件之后的记录", "FUTURE_MODEL")
        if (
            digest(data.feature_columns) != self.feature_spec_hash
            or self._prefix_hash(ids, times, Y, exog, self.train_count) != self.training_prefix_hash
        ):
            raise WorkbenchError("训练期原始事件或观测已改变", "SOURCE_CHANGED")
        key = digest({"events": events, "prefix": self._prefix_hash(ids, times, Y, exog, max(query) + 1)})
        if self._last_prediction and self._last_prediction[0] == key:
            result = self._last_prediction[1]
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in result.items()}
        means = {}
        deviations = {}
        if self.method_id == "VAR":
            history = self.var_history.copy()
            last = self.last_observed.copy()
            wanted = set(query)
            covariance = self.models[0].forecast_cov(steps=1)[0]
            std = np.sqrt(np.maximum(0.0, np.diag(covariance)))
            for index in range(self.last_training_index, max(query) + 1):
                if index > self.last_training_index:
                    observation = (Y[index] - self.mean_) / self.scale_
                    good = np.isfinite(observation)
                    last[good] = observation[good]
                    history = np.vstack((history, last))[-self.models[0].k_ar :]
                if index in wanted:
                    means[index] = self.mean_ + self.models[0].forecast(history, steps=1)[0] * self.scale_
                    deviations[index] = std * self.scale_
        else:
            end = max(query) + 1
            first = min(query) + 1
            normalized = (Y[:end] - self.mean_) / self.scale_
            regressors = None
            extra = {}
            if self.method_id == "SARIMAX":
                lagged = np.vstack((np.full((1, 2), np.nan), exog[: end - 1]))
                regressors = np.column_stack(
                    (np.ones(end), self.exog_scale.transform(self.exog_imputer.transform(lagged)))
                )
                extra["exog"] = np.column_stack(
                    (np.ones(1), self.exog_scale.transform(self.exog_imputer.transform(exog[end - 1 : end])))
                )
            target_means = []
            target_std = []
            for target, result in enumerate(self.models):
                model = (
                    result.model.clone(normalized[:, target], exog=regressors)
                    if regressors is not None
                    else result.model.clone(normalized[:, target])
                )
                filtered = model.filter(result.params)
                prediction = filtered.get_prediction(start=first, end=end, information_set="predicted", **extra)
                target_means.append(np.asarray(prediction.predicted_mean))
                target_std.append(np.sqrt(np.maximum(0.0, np.asarray(prediction.var_pred_mean))))
            for index in set(query):
                position = index + 1 - first
                means[index] = self.mean_ + np.array([v[position] for v in target_means]) * self.scale_
                deviations[index] = np.array([v[position] for v in target_std]) * self.scale_
        output = {
            "mean": np.vstack([means[i] for i in query]),
            "std": np.vstack([deviations[i] for i in query]),
            "kind": "marginal_standard_deviation",
            "calibrated": False,
            "joint_region": False,
            "state_updates_only": True,
            "event_steps_preserved": True,
        }
        if not np.isfinite(output["mean"]).all() or not np.isfinite(output["std"]).all():
            raise WorkbenchError("序列预测出现非有限值", "MODEL_OUTPUT_VALUES")
        self._last_prediction = (key, {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in output.items()})
        return output
