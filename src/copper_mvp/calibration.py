"""Independent temporal split conformal intervals in physical target units."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from copper_mvp.common import WorkbenchError

CALIBRATION_VERSION = "g6n.split-c90.v1"


def finite_sample_quantile(scores, coverage=0.9):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise WorkbenchError("校准分数必须为非空有限向量", "CALIBRATION_SCORES")
    if not 0 < coverage < 1:
        raise WorkbenchError("校准覆盖水平无效", "CALIBRATION_LEVEL")
    rank = math.ceil((len(values) + 1) * coverage)
    if rank > len(values):
        raise WorkbenchError("校准样本不足以形成有限区间", "INSUFFICIENT_CALIBRATION")
    return float(np.partition(values, rank - 1)[rank - 1])


def training_scale(X, y):
    """Scale is fixed by fit data only; zero variation disables joint normalization."""
    values = np.asarray(y, float) - np.asarray(X, float)[:, :2]
    scale = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros(2)
    return np.asarray(scale, float)


@dataclass
class SplitC90:
    coverage: float
    calibration_count: int
    marginal_radius: np.ndarray
    scale: np.ndarray
    joint_radius: float | None

    @classmethod
    def fit(cls, predictions, actual, scale, coverage=0.9, minimum=60):
        point = np.asarray(predictions, float)
        y = np.asarray(actual, float)
        scale = np.asarray(scale, float)
        if point.ndim != 2 or point.shape[1] != 2 or y.shape != point.shape or not np.isfinite([point, y]).all():
            raise WorkbenchError("校准输入必须为有限的双目标矩阵", "CALIBRATION_SHAPE")
        if len(point) < minimum:
            raise WorkbenchError("独立校准段样本不足", "INSUFFICIENT_CALIBRATION")
        if scale.shape != (2,) or not np.isfinite(scale).all() or (scale < 0).any():
            raise WorkbenchError("训练尺度无效", "CALIBRATION_SCALE")
        residual = np.abs(y - point)
        marginal = np.array([finite_sample_quantile(residual[:, j], coverage) for j in range(2)])
        joint = finite_sample_quantile(np.max(residual / scale, axis=1), coverage) if (scale > 0).all() else None
        return cls(coverage, len(point), marginal, scale, joint)

    def intervals(self, predictions):
        point = np.asarray(predictions, float)
        if point.ndim != 2 or point.shape[1] != 2 or not np.isfinite(point).all():
            raise WorkbenchError("区间推理需要有限的双目标点预测", "CALIBRATION_SHAPE")
        results = {"marginal_c90": np.stack([point - self.marginal_radius, point + self.marginal_radius], axis=1)}
        if self.joint_radius is not None:
            radius = self.joint_radius * self.scale
            results["joint_c90"] = np.stack([point - radius, point + radius], axis=1)
        return results

    def manifest(self):
        return {
            "schema_version": CALIBRATION_VERSION,
            "nominal_coverage": self.coverage,
            "calibration_count": self.calibration_count,
            "marginal_radius": self.marginal_radius.tolist(),
            "training_delta_std": self.scale.tolist(),
            "joint_score_radius": self.joint_radius,
            "joint_unavailable_reason": "ZERO_TRAINING_SCALE" if self.joint_radius is None else None,
            "quantile_rule": "ceil((n+1)*coverage)-th order statistic",
            "marginal_is_joint": False,
            "interval_support": "unclipped_real_line",
            "guarantee": "empirical_time_split_evaluation; exchangeability_not_assumed_for_this_process",
        }

    @classmethod
    def from_manifest(cls, value):
        if value["schema_version"] != CALIBRATION_VERSION:
            raise WorkbenchError("校准器版本不兼容", "CALIBRATION_VERSION")
        return cls(
            value["nominal_coverage"],
            value["calibration_count"],
            np.asarray(value["marginal_radius"]),
            np.asarray(value["training_delta_std"]),
            value["joint_score_radius"],
        )


def interval_metrics(actual, point, lower, upper, coverage=0.9, scale=None):
    y, p, lo, hi = (np.asarray(value, float) for value in (actual, point, lower, upper))
    if not (y.shape == p.shape == lo.shape == hi.shape) or not y.size:
        raise WorkbenchError("区间评价维度不一致", "INTERVAL_SHAPE")
    if not np.isfinite([y, p, lo, hi]).all() or (lo > hi).any():
        raise WorkbenchError("区间含非法值或上下界交叉", "INTERVAL_VALUES")
    alpha = 1 - coverage
    width = hi - lo
    score = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    wis = (0.5 * np.abs(y - p) + (alpha / 2) * score) / 1.5
    result = {
        "n": len(y),
        "nominal_coverage": coverage,
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(width.mean()),
        "interval_score": float(score.mean()),
        "wis": float(wis.mean()),
        "negative_lower_bounds": int((lo < 0).sum()),
    }
    result["normalized_width"] = float(width.mean() / scale) if scale is not None and scale > 0 else None
    return result
