"""Run-scoped numerical tools. Only derived summaries leave this module.

Uniform probes diagnose the saved model within the saved search box. Their
counts are never described as the optimizer's original candidate history.
"""

from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from copper_mvp.common import WorkbenchError, safe
from copper_mvp.contracts import RunRequest
from copper_mvp.data import DataRepository
from copper_mvp.modeling import ModelManager
from copper_mvp.optimization import nondominated
from copper_mvp.optimization_problem import positive_scale


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: str = Field(min_length=1, max_length=240, description="简述本次检查要解决什么问题")


class ProbeArguments(ToolArguments):
    points_per_axis: Literal[9, 17, 33] = 17


class ResponseArguments(ProbeArguments):
    axis: Literal["joint", "stage3", "stage4"] = "joint"


TOOL_ARGUMENTS = {
    "inspect_run": ToolArguments,
    "probe_constraints": ProbeArguments,
    "probe_response": ResponseArguments,
    "compare_probe_resolution": ToolArguments,
}
TOOL_DESCRIPTIONS = {
    "inspect_run": "读取已完成优化的模型、原搜索计数、终止原因和相对电流范围；不执行新优化。",
    "probe_constraints": (
        "在原范围内均匀探测9/17/33个点每维，计算As/非负约束排除率和探测前沿。计数属于新探测，不是原NSGA-II轨迹。"
    ),
    "probe_response": (
        "检查固定历史下Cu/As模型响应平台与功率跨度；joint扫描联合网格，stag"
        "e3/stage4扫描单轴。返回聚合统计，不返回原始行或绝对电流。"
    ),
    ("compare_probe_resolution"): (
        "比较9、17、33点每维的有限网格结果，检查探测分辨率影响；不证明全局收敛，不改变约束或重新训练。"
    ),
}


class DiagnosticTools:
    def __init__(self, source: dict, data: DataRepository, models: ModelManager):
        result = source.get("result") or {}
        if source["status"] != "completed" or result.get("kind") != "optimization" or result.get("mode") != "plant":
            raise WorkbenchError("请选择已完成的工厂优化运行", "DIAGNOSIS_SOURCE")
        if not result.get("audit", {}).get("passed"):
            raise WorkbenchError("原优化结果未通过数值复核", "DIAGNOSIS_SOURCE")
        self.source = source
        self.result = result
        self.data = data
        self.models = models
        self.request = RunRequest.model_validate(source["request"])
        self.resolved = models.resolve(
            self.request.event_id, self.request.model_profile, self.request.model_scope, result["bundle_id"]
        )
        if self.resolved["names"] != result["models"] or self.resolved["fold_id"] != result["fold_id"]:
            raise WorkbenchError("原运行与当前解析模型不一致", "MODEL_VERSION_MISMATCH")
        self.stages = result["supported_stages"]
        self.base = np.array([result["reference"]["variables"][f"stage{s}_current_a"] for s in self.stages])
        row = data.row(self.request.event_id)
        self.voltages = np.array([row[f"stage{s}_voltage_v__t_minus_0h"] for s in self.stages])
        self.ranges = {r["stage"]: r for r in result["ranges"]}
        self.variable_stages = [s for s in self.stages if self.ranges.get(s, {}).get("varies", False)]
        matrix = data.candidate_matrix(self.request.event_id, self.stages, self.base[None, :])
        self.reference_y = models.predict_matrix(self.resolved, matrix)[0]
        expected = [result["reference"]["f1"], result["reference"]["as"]]
        if not np.allclose(self.reference_y, expected, rtol=1e-9, atol=1e-8) or not np.isclose(
            self.base @ self.voltages / 1000, result["reference"]["f2"]
        ):
            raise WorkbenchError("原参考点复算不一致", "DIAGNOSIS_REFERENCE")
        fold = None if result["fold_id"] == "DEVELOPMENT" else result["fold_id"]
        _, y = data.training_data()
        self.scale = positive_scale(y.loc[data.training_ids(fold)].to_numpy(float))
        self._probes: dict[tuple, dict] = {}

    def dispatch(self, name: str, arguments: dict) -> tuple[dict, str]:
        if name not in TOOL_ARGUMENTS:
            raise WorkbenchError("没有该诊断工具", "UNKNOWN_TOOL")
        parsed = TOOL_ARGUMENTS[name].model_validate(arguments)
        params = parsed.model_dump(exclude={"purpose"})
        return safe(getattr(self, name)(**params)), parsed.purpose

    def inspect_run(self) -> dict:
        ranges = []
        for stage, base in zip(self.stages, self.base, strict=True):
            r = self.ranges.get(stage)
            ranges.append(
                {
                    "stage": stage,
                    "quantity": "current",
                    "quantity_label": "电流",
                    "unit": "A",
                    "varies": stage in self.variable_stages,
                    "lower_change_pct": 100 * ((r["lower"] if r else base) / base - 1),
                    "upper_change_pct": 100 * ((r["upper"] if r else base) / base - 1),
                    "width_pct_of_reference": 100 * (r["upper"] - r["lower"]) / base if r else 0,
                    "reference_inside_range": bool(r is None or r["lower"] <= base <= r["upper"]),
                    "history_count": r["historical_n"] if r else None,
                }
            )
        return {
            "evidence_scope": "original_saved_run",
            "models": self.result["models"],
            "model_scope": self.result["model_scope"],
            "front_points": self.result["total_front_points"],
            "search_evaluations": self.result["search_evaluations"],
            "feasible_evaluations": self.result["feasible_evaluations"],
            "feasible_rate": self.result["feasible_rate"],
            "stop_reason": self.result["stop_reason"],
            "configured_evaluation_budget": self.request.evaluation_budget,
            "variable_count": len(self.variable_stages),
            "ranges_relative_to_reference": ranges,
            "fixed_quantities": ["电压", "历史锚点", "当前化验浓度", "温度", "流量", "工况"],
            "as_allowance_mg_l": self.request.epsilon_as,
            "as_constraint_definition": (
                "候选As预测 <= 同一模型的参考As预测 + epsilon_As；as_allowance_mg_l"
                "就是epsilon_As。0表示不允许增加，不表示As浓度为0。"
            ),
            "original_rejection_breakdown_available": False,
            "interpretation": (
                "原运行只保存总可行率，未保存逐候选拒绝原因；后续探测只能提供新证据"
                "。电压和历史固定，单个前沿点本身不是故障。"
            ),
        }

    def evaluate_currents(self, currents: np.ndarray) -> dict:
        matrix = self.data.candidate_matrix(self.request.event_id, self.stages, currents)
        y = self.models.predict_matrix(self.resolved, matrix)
        power = currents @ self.voltages / 1000
        F = np.column_stack((y[:, 0], power))
        G = np.column_stack(
            (
                (y[:, 1] - self.reference_y[1] - self.request.epsilon_as) / self.scale[1],
                -y[:, 0] / self.scale[0],
                -y[:, 1] / self.scale[1],
            )
        )
        if not np.isfinite(F).all() or not np.isfinite(G).all():
            raise WorkbenchError("诊断探测出现非法数值", "NONFINITE_DIAGNOSIS")
        feasible = np.all(G <= 1e-8, axis=1)
        indices = np.flatnonzero(feasible)
        front = indices[nondominated(F[indices])] if len(indices) else indices
        front_count = len(np.unique(np.round(F[front], 10), axis=0))
        return {"y": y, "F": F, "G": G, "feasible": feasible, "front_count": front_count}

    def _probe(self, points_per_axis: int, axis: str = "joint") -> dict:
        key = (points_per_axis, axis)
        if key not in self._probes:
            selected = self.variable_stages if axis == "joint" else [int(axis[-1])]
            if any(s not in self.variable_stages for s in selected):
                raise WorkbenchError("所选电流轴在原运行中固定或不可用", "AXIS_NOT_VARIABLE")
            axes = []
            for stage, base in zip(self.stages, self.base, strict=True):
                r = self.ranges.get(stage)
                if stage in selected:
                    values = np.linspace(r["lower"], r["upper"], points_per_axis)
                    if r["lower"] <= base <= r["upper"]:
                        values = np.unique(np.append(values, base))
                else:
                    if r and not r["lower"] <= base <= r["upper"]:
                        raise WorkbenchError(
                            "单轴探测的固定参考电流在原范围外，请使用joint联合探测", "REFERENCE_OUTSIDE_RANGE"
                        )
                    values = np.array([base])
                axes.append(values)
            currents = np.array(list(product(*axes)), dtype=float)
            self._probes[key] = self.evaluate_currents(currents)
        return self._probes[key]

    def probe_constraints(self, points_per_axis: int = 17) -> dict:
        p = self._probe(points_per_axis)
        violated = p["G"] > 1e-8
        as_only = violated[:, 0] & ~violated[:, 1:].any(axis=1)
        return {
            "evidence_scope": "new_uniform_grid_probe",
            "points_per_axis": points_per_axis,
            "probe_points": len(p["F"]),
            "feasible_points": int(p["feasible"].sum()),
            "feasible_rate": float(p["feasible"].mean()),
            "probe_front_points": p["front_count"],
            "rejected_by_as": int(violated[:, 0].sum()),
            "rejected_only_by_as": int(as_only.sum()),
            "negative_cu": int(violated[:, 1].sum()),
            "negative_as": int(violated[:, 2].sum()),
            "as_margin_range_mg_l": [
                float(v)
                for v in [
                    np.min(self.reference_y[1] + self.request.epsilon_as - p["y"][:, 1]),
                    np.max(self.reference_y[1] + self.request.epsilon_as - p["y"][:, 1]),
                ]
            ],
            "interpretation": "各拒绝原因可重叠；探测保持原约束。有限网格不能代表所有连续点或原优化器的拒绝轨迹。",
        }

    def probe_response(self, points_per_axis: int = 17, axis: str = "joint") -> dict:
        p = self._probe(points_per_axis, axis)
        y = p["y"]
        feasible_y = y[p["feasible"]]
        feasible_f = p["F"][p["feasible"]]
        alignment = None
        if len(feasible_f):
            # Compare both objectives at the same feasible point; marginal spans
            # alone cannot distinguish a trade-off from aligned improvement.
            best = feasible_f[np.lexsort((feasible_f[:, 0], feasible_f[:, 1]))[0]]
            dominates = np.all(best <= feasible_f + 1e-8, axis=1) & np.any(best < feasible_f - 1e-8, axis=1)
            alignment = {
                "minimum_power_point_also_minimizes_cu": bool(best[0] <= feasible_f[:, 0].min() + 1e-8),
                "min_power_point_cu_above_probe_min_g_l": float(best[0] - feasible_f[:, 0].min()),
                "points_dominated_by_min_power_point": int(dominates.sum()),
                "feasible_probe_points": len(feasible_f),
            }

        def describe(values):
            if not len(values):
                return {"unique_cu_rounded_8dp": 0, "cu_span_g_l": None, "as_span_mg_l": None}
            return {
                "unique_cu_rounded_8dp": len(np.unique(np.round(values[:, 0], 8))),
                "cu_span_g_l": float(np.ptp(values[:, 0])),
                "as_span_mg_l": float(np.ptp(values[:, 1])),
            }

        return {
            "evidence_scope": "new_uniform_grid_probe",
            "axis": axis,
            "points_per_axis": points_per_axis,
            "probe_points": len(y),
            "all_points": describe(y),
            "feasible_points": describe(feasible_y),
            "power_span_kw": float(np.ptp(p["F"][:, 1])),
            "probe_front_points": p["front_count"],
            "objective_alignment_in_feasible_probe": alignment,
            "cu_constant_in_probe": bool(np.ptp(y[:, 0]) <= 1e-8),
            "as_constant_in_probe": bool(np.ptp(y[:, 1]) <= 1e-8),
            "interpretation": (
                "平台只描述这些探测点上的模型响应。一个点同时最小化两目标可以解释探"
                "测前沿仅有一点；可行点很多时不能说可行域塌缩。响应取值少本身不证明"
                "平台导致单点前沿，也不证明连续区域或实际工艺不敏感。"
            ),
        }

    def compare_probe_resolution(self) -> dict:
        rows = []
        for size in (9, 17, 33):
            p = self._probe(size)
            rows.append(
                {
                    "points_per_axis": size,
                    "probe_points": len(p["F"]),
                    "feasible_rate": float(p["feasible"].mean()),
                    "probe_front_points": p["front_count"],
                    "cu_span_g_l": float(np.ptp(p["y"][:, 0])),
                }
            )
        return {
            "evidence_scope": "new_uniform_grid_probe",
            "resolutions": rows,
            "interpretation": "这是固定范围内三种网格密度的对照。相同前沿点数不等于已证明NSGA-II全局收敛。",
        }
