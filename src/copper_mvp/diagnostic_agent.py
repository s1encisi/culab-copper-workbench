"""DeepSeek tool-calling loop for one saved optimization; no numerical writes."""

from __future__ import annotations

import io
import json
import os
import re
from datetime import UTC, datetime, timedelta, timezone
from time import perf_counter
from typing import Literal

import httpx
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from copper_mas.llm.client import CallBudget, ExternalLLMBudgetExceeded
from copper_mvp.common import PROJECT_ROOT, WORKSPACE_ROOT, WorkbenchError, digest, dumps, utc_now
from copper_mvp.diagnostic_tools import TOOL_ARGUMENTS, TOOL_DESCRIPTIONS, DiagnosticTools

AGENT_VERSION = "optimization-diagnosis-v1.3"
ENDPOINT = "https://api.deepseek.com/chat/completions"
NextCheck = Literal[
    "refine_original_grid", "inspect_tree_partitions", "review_training_support", "review_objective_definition"
]
NEXT_CHECK_LABELS = {
    "refine_original_grid": "在原电流范围和原约束内细化探测，检查采样间隙中的目标权衡。",
    "inspect_tree_partitions": "检查当前冻结树模型在原范围内的分区边界，核对响应平台。",
    "review_training_support": "查看原训练折对当前工况和原电流范围的样本覆盖。",
    "review_objective_definition": "复核当前Cu目标和功率代理的工艺含义、单位与计量覆盖。",
}
QUESTIONS = {
    "why_few_candidates": "为什么这次优化的Pareto候选较少？请区分搜索范围、As约束、模型响应平台和搜索分辨率的证据。",
    "constraint_effect": "As约束对这次优化候选数量的影响有多大？请用原范围内的数值探测说明，保持原约束。",
    "model_response": "这次优化中Cu/As模型对电流变化是否敏感？请检查响应平台及其与功率权衡的关系。",
}
SYSTEM_PROMPT = (
    "你是铜电积离线研究助手，诊断一份已经完成的优化结果。你可以自主选择"
    "本地工具和探测密度，根据工具结果决定下一步，最后调用finish_diagnos"
    "is交付中文报告。\n"
    "本任务的决策变量是三段/四段电流I3/I4（A）；电压固定。ranges_relati"
    "ve_to_reference描述的是电流的相对范围。准确使用工具中的quantity与u"
    "nit，不得把电流扫描写成电压扫描。摘要直接说明诊断发现，基础配置由i"
    "nspect_run证据提供，无需在摘要重复整套配置。\n"
    "先了解原运行，再选择有区分力的数值检查。工具的purpose用一句中文解"
    "释本次检查的目的。至少获取inspect_run和一项数值探测证据；不必机械"
    "执行所有工具。根据用户问题和已返回证据选择工具。\n"
    "结果里evidence_id是可引用的证据编号。每条finding必须引用实际取得的"
    "编号。finish_diagnosis必须单独调用，并且只能在读到工具结果后调用。"
    "不要在正文直接返回报告。\n"
    "区分original_saved_run和new_uniform_grid_probe；后者不是原NSGA-II"
    "候选拒绝轨迹。不得声称已经证明全局最优或收敛。不得把条件模型响应说"
    "成实际工艺的因果效应。单个Pareto点可能是合理的支配结果。\n"
    "准确使用变量：as_allowance_mg_l是允许的As预测增量epsilon_As。epsil"
    "on_As=0意味着候选As预测不高于参考预测，绝不是As浓度为0。\n"
    "准确区分集合：可行点数多但非支配点只有1个，表示探测前沿只有1点，可"
    "行域并未塌缩。优先根据同一点的两个目标支配关系解释单点前沿；预测值"
    "种类较少只是模型响应特征，不能单独证明平台是主因。判断始终限定在已"
    "探测的原范围内。\n"
    "分辨率对照只支持已检查的9/17/33网格结果一致，不足以排除所有采样或"
    "搜索不足；未检查更宽范围，不能排除范围选择的影响。没有对应对照的判"
    "断使用inconclusive。\n"
    "表述分辨率时写“已检查的三种密度均返回1点”；不要写“不是采样伪影”或“"
    "搜索分辨率不是原因”这类已经完全排除的结论。\n"
    "后续检查围绕原范围内的证据缺口。不要为了制造冲突而更换优化目标，不"
    "建议直接扩到30%或50%的范围，不建议对分段常数的DeltaHGB使用普通梯度"
    "优化。诊断结论不需要把所有因素都强行归为确定主因。\n"
    "finish_diagnosis的next_checks从给定动作编号中选择，也可为空；它们"
    "只对应原范围网格加密、当前模型分区检查、原训练折覆盖审查及当前目标"
    "定义复核。\n"
    "固定已有模型、历史背景、范围和As约束；没有训练、改约束或设备控制工"
    "具。发现证据不足时明确说明缺少的检查，不编造现场操作原因。数值和结"
    "论只依据工具返回。报告保持简洁，说明主因判断、依据和仍不确定的部分"
    "。"
)


class Pricing(BaseModel):
    verified_on: str
    source: str
    cache_hit_per_million_cny: float = Field(ge=0)
    cache_miss_per_million_cny: float = Field(gt=0)
    output_per_million_cny: float = Field(gt=0)
    off_peak_multiplier: float = Field(gt=0, le=1)


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["deepseek-v4-flash"]
    thinking: Literal["enabled", "disabled"]
    reasoning_effort: Literal["low", "high", "max"]
    max_calls: int = Field(ge=2, le=10)
    max_tool_calls: int = Field(ge=2, le=16)
    max_output_tokens: int = Field(ge=512, le=16384)
    request_timeout_seconds: int = Field(ge=5, le=120)
    max_wall_seconds: int = Field(ge=30, le=600)
    max_cost_cny: float = Field(gt=0, le=5)
    monthly_limit_cny: float = Field(gt=0)
    pricing: Pricing

    @classmethod
    def load(cls):
        return cls.model_validate(
            yaml.safe_load((PROJECT_ROOT / "configs/llm/diagnostic_agent.yaml").read_text(encoding="utf-8"))
        )


def read_deepseek_key() -> SecretStr | None:
    """Read only DEEPSEEK_API_KEY; do not load the .env into process globals."""
    value = os.environ.get("DEEPSEEK_API_KEY")
    if value and value.strip():
        return SecretStr(value.strip())
    for path in (PROJECT_ROOT / ".env", WORKSPACE_ROOT / ".env"):
        if path.is_file():
            with path.open(encoding="utf-8-sig") as stream:
                for line in stream:
                    if re.match(r"^\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=", line):
                        value = dotenv_values(stream=io.StringIO(line), interpolate=False).get("DEEPSEEK_API_KEY")
                        if value and value.strip():
                            return SecretStr(value.strip())
    return None


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1, max_length=700)
    status: Literal["supported", "inconclusive"]
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class DiagnosisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=1800)
    findings: list[Finding] = Field(min_length=1, max_length=6)
    next_checks: list[NextCheck] = Field(max_length=4)


def tool_definitions() -> list[dict]:
    tools = [
        {
            "type": "function",
            "function": {"name": name, "description": TOOL_DESCRIPTIONS[name], "parameters": model.model_json_schema()},
        }
        for name, model in TOOL_ARGUMENTS.items()
    ]
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "finish_diagnosis",
                "description": (
                    "在读到证据后单独提交最终报告。每项判断引用工具返回的evidence_id，证据不足标inconclusive。"
                ),
                "parameters": DiagnosisReport.model_json_schema(),
            },
        }
    )
    return tools


def estimate_usage(usage: dict, pricing: Pricing, requested_at: datetime) -> dict:
    required = ("prompt_tokens", "completion_tokens")
    if not all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in required):
        raise WorkbenchError("服务端没有提供有效用量，保留费用预留", "LLM_USAGE_MISSING")
    prompt, completion = (usage[k] for k in required)
    hit = usage.get("prompt_cache_hit_tokens", 0)
    if not isinstance(hit, int) or hit < 0 or hit > prompt:
        raise WorkbenchError("服务端缓存用量无效，保留费用预留", "LLM_USAGE_INVALID")
    local = requested_at.astimezone(timezone(timedelta(hours=8)))
    hour = local.hour + local.minute / 60
    peak = local.weekday() < 5 and (9 <= hour < 12 or 14 <= hour < 18)
    factor = 1 if peak else pricing.off_peak_multiplier
    cost = (
        factor
        * (
            hit * pricing.cache_hit_per_million_cny
            + (prompt - hit) * pricing.cache_miss_per_million_cny
            + completion * pricing.output_per_million_cny
        )
        / 1_000_000
    )
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "cache_hit_tokens": hit,
        "estimated_cost_cny": cost,
        "price_period": "peak" if peak else "off_peak",
    }


class DiagnosticAgentService:
    def __init__(self, store, data, models, settings: AgentSettings | None = None, transport=None):
        self.store, self.data, self.models = store, data, models
        self.settings = settings or AgentSettings.load()
        self.transport = transport

    def availability(self) -> dict:
        return {
            "configured": read_deepseek_key() is not None,
            "version": AGENT_VERSION,
            **self.settings.model_dump(exclude={"pricing"}),
            "questions": QUESTIONS,
        }

    def execute(self, run_id: str):
        start = perf_counter()
        self.store.start(run_id)
        job = self.store.get(run_id)
        settings = self.settings
        budget = CallBudget(settings.max_calls, settings.max_cost_cny)
        record = {
            "kind": "agent_diagnosis",
            "source_run_id": job["request"]["source_run_id"],
            "agent_version": AGENT_VERSION,
            "model": settings.model,
            "configuration": settings.model_dump(exclude={"pricing"}),
            "question": job["request"]["question"],
            "report": None,
            "evidence": [],
            "steps": [],
            "calls": [],
            "usage": {
                "api_calls": 0,
                "tool_calls": 0,
                "tool_cache_hits": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_hit_tokens": 0,
                "estimated_cost_cny": 0.0,
                "unsettled_reservation_cny": 0.0,
                "retries": 0,
            },
            "outbound_policy": "DERIVED_SUMMARIES_ONLY",
            "pricing": settings.pricing.model_dump(),
            "concurrency": 1,
        }
        try:
            key = read_deepseek_key()
            if key is None:
                raise WorkbenchError("未找到DEEPSEEK_API_KEY，请在本机.env或环境变量中配置", "LLM_KEY_MISSING")
            source = self.store.get(record["source_run_id"])
            tools = DiagnosticTools(source, self.data, self.models)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": QUESTIONS[record["question"]]},
            ]
            cache = {}
            for round_index in range(settings.max_calls):
                remaining = settings.max_wall_seconds - (perf_counter() - start)
                if remaining <= 0:
                    raise WorkbenchError("诊断时间预算已用完，已保存已取得的证据", "AGENT_TIME_LIMIT")
                payload = {
                    "model": settings.model,
                    "messages": messages,
                    "tools": tool_definitions(),
                    "tool_choice": "auto",
                    "max_tokens": settings.max_output_tokens,
                    "thinking": {"type": settings.thinking},
                    "stream": False,
                }
                if settings.thinking == "enabled":
                    payload["reasoning_effort"] = settings.reasoning_effort
                else:
                    payload["temperature"] = 0
                input_upper = len(dumps(payload).encode("utf-8"))
                reservation = (
                    input_upper * settings.pricing.cache_miss_per_million_cny
                    + settings.max_output_tokens * settings.pricing.output_per_million_cny
                ) / 1_000_000
                reservation_id = budget.reserve(provider_name="deepseek", max_possible_cost_cny=reservation)
                call_id = f"agent:{run_id}:{round_index}"
                requested_at = datetime.now(UTC)
                month = requested_at.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m")
                try:
                    self.store.reserve_cost(call_id, month, reservation, settings.monthly_limit_cny)
                except WorkbenchError:
                    # The monthly check failed before any network request was sent.
                    budget.settle(reservation_id, 0)
                    raise
                call_start = perf_counter()
                record["usage"]["api_calls"] += 1
                self.store.trace(
                    run_id,
                    f"DeepSeek · 第{round_index + 1}轮",
                    "running",
                    utc_now(),
                    0,
                    {"purpose": "根据问题和已取得证据选择下一项工具"},
                )
                try:
                    with httpx.Client(
                        transport=self.transport,
                        timeout=min(settings.request_timeout_seconds, remaining),
                        follow_redirects=False,
                    ) as client:
                        response = client.post(
                            ENDPOINT, headers={"Authorization": "Bearer " + key.get_secret_value()}, json=payload
                        )
                        response.raise_for_status()
                        reply = response.json()
                except httpx.HTTPStatusError as exc:
                    raise WorkbenchError(
                        f"DeepSeek返回HTTP {exc.response.status_code}，本次未自动重试", "LLM_HTTP_ERROR"
                    ) from None
                except (httpx.RequestError, ValueError):
                    raise WorkbenchError(
                        "DeepSeek请求失败或返回格式错误，本次未自动重试", "LLM_REQUEST_ERROR"
                    ) from None
                measure = estimate_usage(reply.get("usage") or {}, settings.pricing, requested_at)
                self.store.settle_cost(call_id, measure["estimated_cost_cny"])
                budget.settle(reservation_id, measure["estimated_cost_cny"])
                for field in ("input_tokens", "output_tokens", "cache_hit_tokens", "estimated_cost_cny"):
                    record["usage"][field] += measure[field]
                record["calls"].append(
                    {
                        "round": round_index + 1,
                        "response_model": reply.get("model"),
                        "duration_ms": (perf_counter() - call_start) * 1000,
                        **measure,
                    }
                )
                self.store.trace(
                    run_id,
                    f"DeepSeek · 第{round_index + 1}轮",
                    "completed",
                    requested_at.isoformat(),
                    record["calls"][-1]["duration_ms"],
                    measure,
                )
                choices = reply.get("choices") or []
                if not choices or choices[0].get("finish_reason") == "length":
                    raise WorkbenchError("大模型输出缺失或达到输出长度上限", "LLM_INCOMPLETE")
                message = choices[0].get("message") or {}
                # Preserve reasoning_content for the provider protocol in memory only.
                messages.append(
                    {k: message[k] for k in ("role", "content", "reasoning_content", "tool_calls") if k in message}
                )
                calls = message.get("tool_calls") or []
                if not calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": "请通过工具取得证据，并调用finish_diagnosis提交报告。不要只返回文字。",
                        }
                    )
                for call in calls:
                    function = call.get("function") or {}
                    name = function.get("name", "")
                    clock = perf_counter()
                    purpose = ""
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                        if name == "finish_diagnosis":
                            if len(calls) != 1:
                                raise WorkbenchError(
                                    "请读到工具结果后，单独调用finish_diagnosis", "REPORT_REQUIRES_FEEDBACK"
                                )
                            report = DiagnosisReport.model_validate(arguments)
                            evidence_ids = {e["evidence_id"] for e in record["evidence"]}
                            tool_names = {e["tool"] for e in record["evidence"]}
                            if "inspect_run" not in tool_names or len(tool_names) < 2:
                                raise WorkbenchError("需要原运行摘要及至少一项数值探测证据", "REPORT_MISSING_EVIDENCE")
                            if any(set(f.evidence_ids) - evidence_ids for f in report.findings):
                                raise WorkbenchError("报告引用了不存在的证据编号", "REPORT_UNKNOWN_EVIDENCE")
                            record["report"] = {
                                **report.model_dump(),
                                "next_check_codes": report.next_checks,
                                "next_checks": [NEXT_CHECK_LABELS[key] for key in report.next_checks],
                            }
                            self.store.trace(
                                run_id,
                                "完成有证据的诊断",
                                "completed",
                                utc_now(),
                                0,
                                {"evidence_count": len(evidence_ids)},
                            )
                            self._save(run_id, record, budget, start)
                            self.store.finish(run_id, record, (perf_counter() - start) * 1000)
                            return
                        if record["usage"]["tool_calls"] >= settings.max_tool_calls:
                            raise WorkbenchError("本次工具预算已用完，请根据已有证据结束诊断", "TOOL_LIMIT")
                        record["usage"]["tool_calls"] += 1
                        if name not in TOOL_ARGUMENTS:
                            raise WorkbenchError("没有该诊断工具", "UNKNOWN_TOOL")
                        TOOL_ARGUMENTS[name].model_validate(arguments)
                        cache_key = digest(
                            {"name": name, "arguments": {k: v for k, v in arguments.items() if k != "purpose"}}
                        )
                        if cache_key in cache:
                            result = cache[cache_key]
                            purpose = str(arguments.get("purpose", "复用已有探测"))
                            cached = True
                            record["usage"]["tool_cache_hits"] += 1
                        else:
                            output, purpose = tools.dispatch(name, arguments)
                            result = {"evidence_id": f"E{len(record['evidence']) + 1}", "tool": name, "data": output}
                            record["evidence"].append(result)
                            cache[cache_key] = result
                            cached = False
                        step = {
                            "tool": name,
                            "purpose": purpose,
                            "evidence_id": result["evidence_id"],
                            "cached": cached,
                            "status": "completed",
                            "duration_ms": (perf_counter() - clock) * 1000,
                        }
                    except (ValidationError, json.JSONDecodeError, WorkbenchError, AttributeError, TypeError) as exc:
                        code = getattr(exc, "code", "TOOL_ARGUMENTS")
                        message_text = (
                            str(exc) if isinstance(exc, WorkbenchError) else "工具参数不符合结构，请根据工具定义修正"
                        )
                        result = {"error": {"code": code, "message": message_text}}
                        if isinstance(exc, ValidationError):
                            result["error"]["fields"] = [
                                {"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                                for e in exc.errors(include_input=False, include_context=False, include_url=False)
                            ]
                        step = {
                            "tool": name,
                            "purpose": purpose,
                            "status": "failed",
                            "error": result["error"],
                            "duration_ms": (perf_counter() - clock) * 1000,
                        }
                    record["steps"].append(step)
                    self.store.trace(run_id, name, step["status"], utc_now(), step["duration_ms"], step)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": dumps(result)})
                self._save(run_id, record, budget, start)
            raise WorkbenchError("模型调用预算已用完，尚未形成有效诊断报告；已保存证据", "AGENT_CALL_LIMIT")
        except Exception as exc:
            code = getattr(exc, "code", type(exc).__name__)
            message = (
                str(exc)
                if isinstance(exc, (WorkbenchError, ExternalLLMBudgetExceeded))
                else "诊断执行失败，已保留证据和用量记录"
            )
            record["error"] = {"code": code, "message": message}
            self._save(run_id, record, budget, start)
            self.store.fail(run_id, code, message, (perf_counter() - start) * 1000)

    def _save(self, run_id, record, budget, started):
        record["usage"]["unsettled_reservation_cny"] = budget.reserved_cost_cny
        record["elapsed_ms"] = (perf_counter() - started) * 1000
        self.store.record_result(run_id, record)
