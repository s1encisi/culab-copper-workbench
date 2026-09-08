from __future__ import annotations

import math
import os
import re
import threading
from datetime import datetime, timezone, timedelta

from copper_mas.agents.llm_roles import ExplanationCardV1, build_deepseek_explanation_task
from copper_mas.config.settings import load_provider_registry, resolve_provider_runtime
from copper_mas.llm.client import CallBudget, call_structured
from copper_mvp.common import PROJECT_ROOT, WorkbenchError, digest
from copper_mvp.storage import RunStore


def template(run: dict, question: str) -> dict:
    result = run["result"] or {}
    kind = result.get("kind")
    if kind == "prediction":
        predictions = result["predictions"]
        parts = []
        for target, label in (("cu", "Cu"), ("as", "As")):
            p = predictions[target]
            if p["value"] is not None:
                parts.append(f"{label} 预测为 {p['value']:.3f} {p['unit']}，相对当前值变化 {p['delta']:+.3f}；使用 {p['model']}。")
        if question == "mode":
            names = ("一期一二段", "二期一二段", "三段", "四段")
            parts = ["工况记录：" + "，".join(f"{n} {s}" for n, s in zip(names, result["mode"])) + "。", "状态来自最近可用电流，电压、流量或槽数用于补充证据。UNKNOWN 表示缺少可用电流。"]
        elif question == "model":
            parts.append("Persistence 延续当前浓度；变化量模型利用当前和历史过程字段。模型实验页保留所有五折结果，可比较 MAE、RMSE 与逐折表现。")
        else:
            parts.append("该任务预测下一次录入事件，时间间隔并非固定两小时。")
        if result.get("fold_id"):
            parts.append(f"模型用途：{result['model_scope']}，对应 {result['fold_id']}。")
        if result.get("warnings"):
            messages = []
            for warning in result["warnings"]:
                message = str(warning)
                for code, label in (("P1_STAGE12", "一期一二段"), ("P2_STAGE12", "二期一二段"), ("S3", "三段"), ("S4", "四段")):
                    message = message.replace(code + "_CURRENT_MISSING", label + "缺少可用电流").replace(code + "_SIGNAL_CONFLICT", label + "的电流与辅助信号存在冲突").replace(code + "_COMPANION_MISSING", label + "缺少辅助工况证据")
                message = {"CURRENT_RESULT_QUALITY_WARNING": "当前化验带有来源质量提示", "MODEL_NOT_READY": "本次使用 Persistence 参照", "NO_OOF_MODEL": "该事件未配置折外模型，本次使用 Persistence"}.get(message, message)
                messages.append(message)
            parts.append("本次数据与模型提示：" + "；".join(messages) + "。")
    elif kind == "optimization":
        if not result["candidates"]:
            parts = ["本次预算内没有找到满足配置的候选，参考方案和约束信息已保留。可检查局部范围与 As 容许变化后重新实验。"]
        else:
            candidate = next(c for c in result["candidates"] if c["id"] == result["representatives"]["balanced"])
            if result["mode"] == "plant":
                parts = [f"共找到 {result['total_front_points']} 个非支配候选。平衡方案的 Cu 为 {candidate['f1']:.3f} g/L，电功率代理为 {candidate['f2']:.2f} kW。", f"相对同代理参考点，Cu 降低 {candidate['delta_cu']:.3f} g/L，功率代理降低 {candidate['delta_power']:.2f} kW；As 约束裕度为 {candidate['as_margin']:.3f} mg/L。", "候选使用固定电压与历史上下文，呈现模型空间中的质量—功率权衡。"]
                if question == "constraints":
                    parts.append("As 比较基准是同一个响应模型对参考电流的预测；搜索范围由训练期分位数与局部变化比例取交集。")
            else:
                parts = [f"数学测试得到 {result['total_front_points']} 个候选，平衡点 f₁={candidate['f1']:.4f}、f₂={candidate['f2']:.4f}。", "函数为两组平方距离，约束 u₁+u₂≤3；结果为无量纲求解验证。"]
            parts.append("Cu/目标一优先、功率/目标二优先与平衡方案使用不同偏好，从同一已审计候选集中选取。")
    elif kind == "training":
        parts = [f"完成 {result['train_count']} 条主样本的模型实验，OOF 覆盖 {result['oof_events']} 个事件。", "每个目标比较 Persistence、DeltaRidge 与 DeltaHGB，保存五折模型和全开发期模型。历史回放默认匹配对应时间折。"]
    else:
        parts = [result.get("message", "该运行的结果和节点记录已保存。")]
    return {"mode": "local", "text": "\n\n".join(parts), "sources": [run["run_id"]], "question": question, "input_tokens": 0, "output_tokens": 0, "estimated_cost_cny": 0.0}


class ExplanationService:
    def __init__(self, store: RunStore):
        self.store = store
        self.lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return os.environ.get("COPPER_MVP_LIVE_LLM") == "1"

    def explain(self, run_id: str, question: str, use_llm: bool = False, transport=None) -> dict:
        run = self.store.get(run_id)
        if run["status"] != "completed":
            raise WorkbenchError("请等待任务完成后查看解释", "RUN_NOT_COMPLETE")
        key = digest({"run": run_id, "question": question, "live": use_llm and self.enabled, "version": "explain-v1.1", "model": os.environ.get("DEEPSEEK_MODEL", "")})
        with self.lock:
            existing = self.store.cached_explanation(key)
            if existing:
                return {**existing, "cached": True}
            result = template(run, question)
            if use_llm and self.enabled:
                try:
                    monthly = float(os.environ.get("COPPER_MVP_MONTHLY_CNY", "0"))
                    cap = float(os.environ.get("COPPER_MVP_CALL_CAP_CNY", "0"))
                    input_price = float(os.environ.get("COPPER_MVP_INPUT_CNY_PER_MILLION", "0"))
                    output_price = float(os.environ.get("COPPER_MVP_OUTPUT_CNY_PER_MILLION", "0"))
                    if not all(math.isfinite(v) and v > 0 for v in (monthly, cap, input_price, output_price)) or not os.environ.get("DEEPSEEK_MODEL"):
                        raise WorkbenchError("请先配置已确认的模型、预算和单价", "LLM_CONFIG")
                    registry = load_provider_registry(PROJECT_ROOT / "configs/llm/providers.yaml")
                    env = {**os.environ, "COPPER_MAS_LIVE_CALLS": "true"}
                    runtime = resolve_provider_runtime(registry, "deepseek_executor", environ=env).model_copy(update={"input_cny_per_million_tokens": input_price, "output_cny_per_million_tokens": output_price})
                    values = run["result"]
                    reasons = ["RUN_COMPLETED", "HAS_WARNINGS" if values.get("warnings") else "NO_WARNINGS", "QUESTION_" + question.upper()]
                    task = build_deepseek_explanation_task(artifact_ids=(run_id,), pass_fail_status="PASSED", reason_codes=tuple(reasons), bounded_counts={"task_type": run["task_type"], "candidate_count": len(values.get("candidates", [])), "warning_count": len(values.get("warnings", []))}, explanation_scope="A5_AUDIT_SUMMARY")
                    month = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m")
                    self.store.reserve_cost(key, month, cap, monthly)
                    called = call_structured(runtime=runtime, task=task, output_model=ExplanationCardV1, budget=CallBudget(max_calls=1, max_cost_cny=cap), timeout_seconds=20, transport=transport)
                    self.store.settle_cost(key, called.estimated_cost_cny)
                    if set(called.value.artifact_ids) != {run_id}:
                        raise WorkbenchError("说明引用未匹配本次运行", "EXPLANATION_REFERENCE")
                    summary = called.value.summary
                    if re.search(r"\d", re.sub(r"A[1-5]", "", summary)):
                        raise WorkbenchError("语言说明含未核验数值，使用本地计算说明", "EXPLANATION_NUMERIC")
                    result.update({"mode": "llm", "text": result["text"] + "\n\n语言说明：" + summary, "input_tokens": called.input_tokens, "output_tokens": called.output_tokens, "estimated_cost_cny": called.estimated_cost_cny, "provider_model": called.model})
                except Exception as exc:
                    result["fallback_reason"] = getattr(exc, "code", "EXPLANATION_UNAVAILABLE")
            elif use_llm:
                result["fallback_reason"] = "LIVE_LLM_DISABLED"
            self.store.save_explanation(key, run_id, result)
            return {**result, "cached": False}
