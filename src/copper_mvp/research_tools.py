"""Read-only business tools for free questions; raw event values stay local."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from copper_mvp.access import PROJECT
from copper_mvp.common import APP_VERSION, WorkbenchError, digest, safe
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.diagnostic_tools import DiagnosticTools
from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_evaluation import read_training
from copper_mvp.model_registry import catalog
from copper_mvp.optimizer_registry import optimizer_catalog


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    purpose: str = Field(min_length=1, max_length=200)


class ReferenceArguments(Arguments):
    reference: str = Field(default="selected", min_length=1, max_length=48)


class MetricArguments(ReferenceArguments):
    as_of: AwareDatetime | None = None


class ProbeArguments(ReferenceArguments):
    points_per_axis: Literal[9, 17, 33] = 17
    axis: Literal["joint", "stage3", "stage4"] = "joint"


TOOLS = {
    "project_status": (Arguments, "读取工作台当前能力、方法目录和可用数据范围。"),
    "list_model_comparisons": (Arguments, "列出最近的模型比较，返回可继续查询的匿名引用。"),
    "model_metrics": (MetricArguments, "在指定截止时间，用共同成熟事件重新计算模型误差。"),
    "list_optimizations": (Arguments, "列出已完成工厂优化及优化器比较的匿名引用。"),
    "optimization_summary": (ReferenceArguments, "读取一次优化或优化器比较的统计、约束定义和运行计数。"),
    "probe_constraints": (ProbeArguments, "在所选原优化范围内计算约束与可行集统计。"),
    "probe_response": (ProbeArguments, "在所选原优化范围内检查模型响应和目标权衡。"),
    "probe_resolution": (ReferenceArguments, "对所选原优化比较三种网格密度的结果。"),
    "event_context": (ReferenceArguments, "读取所选事件的工况与输入完整性；具体浓度保留为本地事实引用。"),
}


def definitions():
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": schema.model_json_schema()}} for name, (schema, description) in TOOLS.items()]


class ResearchTools:
    def __init__(self, workbench, principal, context, references=None):
        principal.require("read")
        if principal.project_id != PROJECT:
            raise WorkbenchError("当前项目不可访问工厂资源", "FORBIDDEN")
        self.wb, self.principal, self.context = workbench, principal, context
        self.references = dict(references or {})
        self.models = ComparisonService(workbench.root / "model_comparisons", workbench.data)
        self.probes = {}

    def register(self, kind, identifier):
        alias = kind + "-" + digest(identifier)[:12]
        self.references[alias] = {"kind": kind, "id": identifier}
        return alias

    def resolve(self, reference, kind):
        if reference == "selected":
            field = {"O": "optimization_run_id", "M": "model_comparison_id",
                     "C": "optimizer_comparison_id", "E": "event_id"}[kind]
            identifier = self.context.get(field)
            if identifier:
                self.register(kind, identifier)
                return identifier
            matches = [v["id"] for v in self.references.values() if v["kind"] == kind]
            if len(matches) == 1:
                return matches[0]
            raise WorkbenchError("请先选择一个对应资源，或用列表工具取得引用", "CONTEXT_REQUIRED")
        record = self.references.get(reference)
        if not record or record["kind"] != kind:
            raise WorkbenchError("没有该资源引用，请先读取列表", "RESOURCE_NOT_FOUND")
        return record["id"]

    def execute(self, name, arguments):
        if name not in TOOLS:
            raise WorkbenchError("没有这个只读工具", "UNKNOWN_TOOL")
        parsed = TOOLS[name][0].model_validate(arguments)
        params = parsed.model_dump(exclude={"purpose"}, mode="python")
        payload = getattr(self, name)(**params)
        return {"summary": safe(payload["summary"] if "local_facts" in payload else payload),
                "local_facts": safe(payload.get("local_facts", {})), "_resource_refs": dict(self.references)}

    def project_status(self):
        data = self.wb.data
        return {"version": APP_VERSION, "models": catalog(), "optimizers": optimizer_catalog(),
                "development_events": len(data.frame), "oof_events": len(data.fold_for),
                "data_period": {"start": source_time(data.frame.decision_at.min()).isoformat(),
                                "end": source_time(data.frame.decision_at.max()).isoformat()},
                "available_operations": ["模型指标查询", "优化结果比较", "原范围数值探测", "事件工况查询"],
                "process_values": "observations", "tool_effects": "read_only"}

    def list_model_comparisons(self):
        rows = []
        for state in self.models.list()[:8]:
            if state["status"] == "completed":
                value = self.models.get(state["run_id"])["result"]
                rows.append({"reference": self.register("M", state["run_id"]),
                             "methods": value["methods"], "common_events": value["common_events"],
                             "evaluation_mode": value["evaluation_mode"]})
        return {"items": rows}

    def model_metrics(self, reference="selected", as_of=None):
        identifier = self.resolve(reference, "M")
        record = self.models.get(identifier)["result"]
        root = self.models.directory(identifier)
        _, protocol = read_training(root)
        bound = source_time(self.context["as_of"]) if self.context.get("as_of") else source_time(protocol["evaluation_as_of"])
        cutoff = source_time(as_of) if as_of else bound
        if cutoff > bound:
            raise WorkbenchError("查询截止超出本任务时间范围", "AS_OF_SCOPE")
        ledger = DataService(self.wb.data).labels
        visible = ledger.latest(cutoff)
        frame = pd.read_csv(root / "oof_predictions.csv")
        methods = record["methods"]
        events = {e for e in self.wb.data.fold_for if all((e, t) in visible and visible[(e, t)].quality_eligible
                  and visible[(e, t)].value is not None for t in ("cu", "as"))}
        frames = {}
        for method in methods:
            rows = frame[frame.method_id.eq(method)].set_index("event_id")
            good = rows.status.eq("completed") & np.isfinite(rows[["cu", "as"]].to_numpy(float)).all(axis=1)
            events &= set(rows.index[good])
            frames[method] = rows
        ids = sorted(events)
        metrics = []
        for method in methods:
            for target, unit in (("cu", "g/L"), ("as", "mg/L")):
                actual = np.array([visible[(e, target)].value for e in ids])
                predicted = frames[method].loc[ids, target].to_numpy(float)
                metrics.append({"method": method, "target": target, "unit": unit, "n": len(ids),
                    "mae": float(np.abs(predicted - actual).mean()) if len(ids) else None,
                    "rmse": float(np.sqrt(np.square(predicted - actual).mean())) if len(ids) else None})
        for metric in metrics:
            baseline = next((m["mae"] for m in metrics if m["method"] == "Persistence" and m["target"] == metric["target"]), None)
            metric["mae_difference_vs_persistence"] = metric["mae"] - baseline if metric["mae"] is not None and baseline is not None else None
        return {"reference": self.register("M", identifier), "as_of": cutoff.isoformat(),
                "status": "READY" if len(ids) >= 60 else "INSUFFICIENT_LABELS",
                "common_events": len(ids), "metrics": metrics, "automatic_promotion": False}

    def list_optimizations(self):
        original = []
        for run in self.wb.store.list("optimize", limit=30):
            if run["status"] == "completed" and run["request"].get("mode", "plant") == "plant":
                result = self.wb.store.get(run["run_id"])["result"]
                original.append({"reference": self.register("O", run["run_id"]),
                                 "front_points": result["total_front_points"], "model_scope": result["model_scope"]})
                if len(original) == 6:
                    break
        comparisons = []
        root = self.wb.root / "optimizer_comparisons"
        for path in sorted(root.glob("*/comparison.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:6]:
            result = json.loads(path.read_text(encoding="utf-8"))
            comparisons.append({"reference": self.register("C", path.parent.name),
                                "cases": result["cases"], "optimizers": result["optimizers"], "runs": result["runs"]})
        return {"original_runs": original, "optimizer_comparisons": comparisons}

    def diagnostic(self, reference):
        identifier = self.resolve(reference, "O")
        if identifier not in self.probes:
            run = self.wb.store.get(identifier)
            decision = source_time(run["result"]["decision_at"])
            if self.context.get("as_of") and decision > source_time(self.context["as_of"]):
                raise WorkbenchError("该优化事件晚于任务截止", "AS_OF_SCOPE")
            self.probes[identifier] = DiagnosticTools(run, self.wb.data, self.wb.models)
        return self.probes[identifier]

    def optimization_summary(self, reference="selected"):
        comparison = (reference.startswith("C-") or
                      (reference == "selected" and self.context.get("optimizer_comparison_id") and not self.context.get("optimization_run_id")))
        if not comparison:
            return self.diagnostic(reference).inspect_run()
        identifier = self.resolve(reference, "C")
        path = self.wb.root / "optimizer_comparisons" / identifier / "comparison.json"
        if not path.is_file():
            raise WorkbenchError("没有该优化器比较", "RESOURCE_NOT_FOUND")
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value["results"]
        if self.context.get("as_of"):
            # Compare only instances whose historical event is already known.
            cutoff = source_time(self.context["as_of"])
            rows = [r for r in rows if not r["event_id"] or source_time(self.wb.data.row(r["event_id"]).decision_at) <= cutoff]
        frame = pd.DataFrame(rows)
        summaries = []
        if len(frame):
            baseline = frame[frame.optimizer_id.eq("NSGA-II")].set_index(["case", "seed"])
            for method, part in frame.groupby("optimizer_id"):
                paired = part.set_index(["case", "seed"])
                common = paired.index.intersection(baseline.index)
                difference = paired.loc[common, "hv"].to_numpy() - baseline.loc[common, "hv"].to_numpy()
                summaries.append({"optimizer": method, "runs": len(part),
                    "wins_vs_nsga2": int((difference > 1e-10).sum()),
                    "ties_vs_nsga2": int((np.abs(difference) <= 1e-10).sum()),
                    "losses_vs_nsga2": int((difference < -1e-10).sum()),
                    "median_ms": float(part.elapsed_ms.median()),
                    "median_front_points": float(part.verified_front_points.median()),
                    "max_evaluations": int(part.total_evaluations.max())})
        return {"reference": self.register("C", identifier), "summary": summaries, "comparison_basis": "paired_case_and_seed"}

    def probe_constraints(self, reference="selected", points_per_axis=17, axis="joint"):
        return self.diagnostic(reference).probe_constraints(points_per_axis)

    def probe_response(self, reference="selected", points_per_axis=17, axis="joint"):
        return self.diagnostic(reference).probe_response(points_per_axis, axis)

    def probe_resolution(self, reference="selected"):
        return self.diagnostic(reference).compare_probe_resolution()

    def event_context(self, reference="selected"):
        identifier = self.resolve(reference, "E")
        context = self.wb.data.context(identifier)
        if self.context.get("as_of") and source_time(context["decision_at"]) > source_time(self.context["as_of"]):
            raise WorkbenchError("事件晚于任务截止", "AS_OF_SCOPE")
        return {"summary": {"reference": self.register("E", identifier), "decision_at": source_time(context["decision_at"]).isoformat(),
                   "mode": context["mode"], "admission_status": context["admission_status"],
                   "warnings": context["warnings"], "feature_count": context["feature_count"],
                   "local_fact_names": ["current_cu", "current_as"]},
                "local_facts": {"current_cu": {"value": context["cu"], "unit": "g/L"},
                                "current_as": {"value": context["as"], "unit": "mg/L"}}}


def public_evidence(evidence):
    payload = evidence["data"]
    return {"evidence_id": evidence["evidence_id"], "tool": evidence["tool"],
            "data": payload["summary"], "local_fact_names": list(payload.get("local_facts", {}))}


def render_answer(template, evidence):
    lookup = {e["evidence_id"]: e["data"].get("local_facts", {}) for e in evidence}
    def replace(match):
        identifier, field = match.groups()
        fact = lookup.get(identifier, {}).get(field)
        if fact is None:
            raise WorkbenchError("回答引用了不存在的本地数值", "ANSWER_FACT_REFERENCE")
        return "缺测" if fact["value"] is None else f"{fact['value']:.6g} {fact['unit']}"
    return re.sub(r"\{\{(E-[a-f0-9]+)\.([A-Za-z_]+)\}\}", replace, template)
