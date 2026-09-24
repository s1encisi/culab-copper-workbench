"""Predeclared portfolio comparisons with case-wise pairing and ledger verification."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from threadpoolctl import threadpool_limits

from copper_mvp.common import WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.contracts import RunRequest
from copper_mvp.ensemble_experiment import process_peak_mb
from copper_mvp.model_training import source_signature
from copper_mvp.optimization_problem import build_problem
from copper_mvp.optimizer_comparison import TOLERANCE, optimizer_software
from copper_mvp.optimizer_portfolio import POLICY, STRATEGIES, run_portfolio

PORTFOLIO_FILES = (
    "optimizer_portfolio.py",
    "portfolio_comparison.py",
    "optimization_problem.py",
    "optimizer_comparison.py",
)


class PortfolioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    mode: Literal["plant", "benchmark"] = "plant"
    event_ids: tuple[str, ...] = ()
    strategies: tuple[str, ...] = STRATEGIES
    seeds: tuple[int, ...] = tuple(range(20260911, 20260921))
    model_profile: Literal["DeltaHGB", "DeltaRidge"] = "DeltaHGB"
    model_scope: Literal["oof_replay", "development_analysis"] = "oof_replay"
    bundle_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    radius: float = Field(default=0.1, ge=0.01, le=0.2)
    epsilon_as: float = Field(default=0, ge=0, le=5000)
    total_budget: int = Field(default=2048, ge=128, le=16384)
    seconds_per_run: int = Field(default=120, ge=5, le=600)

    @model_validator(mode="after")
    def valid_scope(self):
        if (
            not self.strategies
            or len(set(self.strategies)) != len(self.strategies)
            or any(s not in STRATEGIES for s in self.strategies)
            or "fixed_nsga2" not in self.strategies
        ):
            raise ValueError("策略必须唯一、受支持，并包含固定 NSGA-II 对照")
        if (
            not 1 <= len(self.seeds) <= 10
            or len(set(self.seeds)) != len(self.seeds)
            or any(s < 0 or s >= 2**31 for s in self.seeds)
        ):
            raise ValueError("使用一到十个不重复种子")
        if len(self.event_ids) > 20 or len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("工况必须唯一且不超过二十个")
        if (self.mode == "plant" and not self.event_ids) or (self.mode == "benchmark" and self.event_ids):
            raise ValueError("历史工况与数学模式的事件范围不匹配")
        return self

    def problem_request(self, event, seed, bundle):
        return RunRequest(
            task_type="optimize",
            request_key=self.request_key,
            mode=self.mode,
            event_id=event,
            model_profile=self.model_profile,
            model_scope=self.model_scope,
            bundle_id=bundle,
            radius=self.radius,
            epsilon_as=self.epsilon_as,
            seed=seed,
            evaluation_budget=64,
        )


def source_record(data, models, request):
    if request.mode == "benchmark":
        return {"kind": "declared_quadratic_benchmark"}, None
    bundle = models.manifest(request.bundle_id)
    dependencies = {}
    for key, entry in bundle["artifacts"].items():
        if f":{request.model_profile}:" in key:
            path = models.root / bundle["bundle_id"] / entry["file"]
            if file_hash(path) != entry["sha256"]:
                raise WorkbenchError("响应模型文件哈希不符", "MODEL_HASH_MISMATCH")
            dependencies[key] = entry["sha256"]
    return {
        "data": source_signature(data),
        "bundle_id": bundle["bundle_id"],
        "model_dependencies": dependencies,
    }, bundle["bundle_id"]


def verify_run(root, result):
    root = Path(root)
    for name, expected in result["evidence_hashes"].items():
        if file_hash(root / name) != expected:
            raise WorkbenchError("优化运行证据哈希不符", "OPTIMIZER_EVIDENCE")
    problem = json.loads((root / "problem.json").read_text(encoding="utf-8"))
    signature = problem.pop("signature")
    if digest(problem) != signature or signature != result["problem_signature"]:
        raise WorkbenchError("问题签名不一致", "OPTIMIZER_SIGNATURE")
    rows = pd.read_csv(root / "evaluations.csv")
    if (
        len(rows) != result["total_evaluations"]
        or sum(result["evaluations"].values()) != len(rows)
        or len(rows) > result["budget_limit"]
    ):
        raise WorkbenchError("求值账本与总预算不一致", "OPTIMIZER_BUDGET")
    for phase, count in result["evaluations"].items():
        if rows.phase.eq(phase).sum() != count:
            raise WorkbenchError("分阶段求值数量不一致", "OPTIMIZER_BUDGET")
    if problem["mode"] == "benchmark":
        X = rows[["x0", "x1"]].to_numpy()
        if not np.allclose(
            rows[["f0", "f1"]], np.column_stack(((X * X).sum(1), ((X - 2) ** 2).sum(1)))
        ) or not np.allclose(rows.g0, X.sum(1) - 3):
            raise WorkbenchError("数学目标或约束复核失败", "CANDIDATE_AUDIT")
    front = np.array([[c["f1"], c["f2"]] for c in result["candidates"]]).reshape(-1, 2)
    normalized = front / np.asarray(result["metric"]["scale"])
    previous = 1.1
    hv = 0.0
    for x, y in normalized[np.argsort(normalized[:, 0])]:
        if x <= 1.1 and y < previous:
            hv += (1.1 - x) * (previous - y)
            previous = y
    if not np.isclose(hv, result["hv"], rtol=1e-10, atol=1e-12):
        raise WorkbenchError("独立二维 HV 复核失败", "CANDIDATE_AUDIT")
    if result["evaluations"]["verification"] != len(result["candidates"]):
        raise WorkbenchError("候选没有逐点复算记录", "CANDIDATE_AUDIT")
    for candidate in result["candidates"]:
        if any(v > TOLERANCE for v in candidate["constraints"]):
            raise WorkbenchError("候选约束不满足", "CANDIDATE_AUDIT")
        if problem["mode"] == "plant":
            fixed = problem["fixed_context"]
            currents = np.array([candidate["all_currents"][f"stage{s}_current_a"] for s in fixed["stages"]])
            if not np.isclose(currents @ np.array(fixed["fixed_voltages"]) / 1000, candidate["f2"]):
                raise WorkbenchError("电功率代理复核失败", "CANDIDATE_AUDIT")
            scale = problem["constraints"]["scales"]
            arsenic = candidate["as_value"]
            expected = [
                (arsenic - problem["reference"]["as"] - problem["constraints"]["epsilon_as_mg_l"]) / scale[1],
                -candidate["f1"] / scale[0],
                -arsenic / scale[1],
            ]
            if not np.allclose(expected, candidate["constraints"]):
                raise WorkbenchError("化学约束复核失败", "CANDIDATE_AUDIT")
    trace = json.loads((root / "allocation_trace.json").read_text(encoding="utf-8"))
    pilots = [r for r in trace if r["stage"] == "arm_pilot"]
    if pilots:
        totals = [sum(p["charged_evaluations"] for p in pilots if p["arm"] == arm) for arm in ("NSGA-II", "SPEA2")]
        if len(set(totals)) != 1 or any(p["archive_points_received"] for p in pilots):
            raise WorkbenchError("pilot 不是等额独立比较", "OPTIMIZER_FAIRNESS")
    return {
        "ledger_verified": True,
        "objectives_checked": "analytical_all_rows"
        if problem["mode"] == "benchmark"
        else "power_and_constraints_on_verified_front",
        "hv_checked": True,
        "pilot_fairness_checked": True,
    }


def paired_summary(results, strategies):
    lookup = {(r["case"], r["seed"], r["strategy"]): r for r in results}
    keys = sorted({(r["case"], r["seed"]) for r in results})
    summary = {}
    for strategy in strategies:
        pairs = []
        wins = ties = losses = zero_gains = failures = 0
        for case, seed in keys:
            current = lookup.get((case, seed, strategy))
            baseline = lookup.get((case, seed, "fixed_nsga2"))
            if (
                not current
                or not baseline
                or current["status"] == "failed"
                or baseline["status"] == "failed"
                or len(current.get("attempts", [])) > 1
                or len(baseline.get("attempts", [])) > 1
            ):
                failures += 1
                continue
            if (
                current["problem_signature"] != baseline["problem_signature"]
                or current["initial_population_hash"] != baseline["initial_population_hash"]
            ):
                raise WorkbenchError("配对运行的问题或初始种群不同", "OPTIMIZER_FAIRNESS")
            difference = current["hv"] - baseline["hv"]
            wins += difference > 1e-12
            losses += difference < -1e-12
            ties += abs(difference) <= 1e-12
            relative = difference / baseline["hv"] if baseline["hv"] > 0 else (0.0 if current["hv"] == 0 else None)
            zero_gains += baseline["hv"] == 0 and current["hv"] > 0
            pairs.append(
                {
                    "case": case,
                    "seed": seed,
                    "relative_hv_gain": relative,
                    "elapsed_ratio": current["elapsed_ms"] / baseline["elapsed_ms"],
                    "evaluation_ratio": current["total_evaluations"] / baseline["total_evaluations"],
                }
            )
        usable = [p for p in pairs if p["relative_hv_gain"] is not None]
        interval = None
        if usable:
            rng = np.random.default_rng(20260911)
            cases = sorted({p["case"] for p in usable})
            draws = []
            for _ in range(1000):
                sample = []
                for case in rng.choice(cases, size=len(cases), replace=True):
                    values = [p["relative_hv_gain"] for p in usable if p["case"] == case]
                    sample.extend(rng.choice(values, size=len(values), replace=True))
                draws.append(float(np.median(sample)))
            interval = np.quantile(draws, [0.025, 0.975]).tolist()
        summary[strategy] = {
            "paired_runs": len(pairs),
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "failed_or_unpaired": failures,
            "zero_baseline_hv_improvements": int(zero_gains),
            "median_relative_hv_gain": float(np.median([p["relative_hv_gain"] for p in usable])) if usable else None,
            "relative_gain_ci95": interval,
            "median_elapsed_ratio": float(np.median([p["elapsed_ratio"] for p in pairs])) if pairs else None,
            "pairs": pairs,
            "aggregation": "within-case-and-seed relative HV, then median; no pooled absolute HV ranking",
        }
    return summary


def compare_portfolios(data, models, request, output, progress=lambda value: None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cpu_started = process_time()
    source, bundle = source_record(data, models, request)
    protocol = {
        "schema_version": "portfolio-comparison.g5d.v1",
        "request": request.model_dump(mode="json"),
        "source": source,
        "effective_bundle_id": bundle,
        "policy": POLICY,
        "software": optimizer_software(),
        "code_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in PORTFOLIO_FILES},
        "initialization": (
            "identical metric pilot and 64 evaluated initial points per case/seed; arms get independent populations"
        ),
        "pilot": "equal charged budget, no archive exchange before pilot completion",
        "comparison": "same total vector-evaluation cap including reference, pilots and independent final verification",
        "order": "strategy order rotated by seed index",
        "external_cache": "disabled; within-run archive reuse separately counted",
        "case_selection": "predeclared development cases; not independent new plant validation",
        "execution_authorized": False,
    }
    if (output / "protocol.json").exists():
        previous = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if digest({k: v for k, v in previous.items() if k != "created_at"}) != digest(protocol):
            raise WorkbenchError("恢复所需的数据、模型、代码或配置已变化", "SOURCE_CHANGED")
        protocol = previous
    else:
        protocol["created_at"] = utc_now()
        write_json(output / "protocol.json", protocol)
    results = []
    with threadpool_limits(limits=1):
        for case, event in enumerate(request.event_ids if request.mode == "plant" else (None,)):
            for seed_index, seed in enumerate(request.seeds):
                order = list(request.strategies)
                offset = seed_index % len(order)
                order = order[offset:] + order[:offset]
                for strategy in order:
                    base = output / f"case_{case}" / str(seed) / strategy
                    base.mkdir(parents=True, exist_ok=True)
                    attempt_file = base / "attempts.json"
                    attempts = json.loads(attempt_file.read_text(encoding="utf-8")) if attempt_file.exists() else []
                    if attempts and attempts[-1]["status"] == "running":
                        interrupted = attempts[-1]
                        path = (output / interrupted["result_path"]).resolve()
                        if not path.is_relative_to(output.resolve()):
                            raise WorkbenchError("运行路径越界", "SOURCE_CHANGED")
                        if path.exists():
                            recovered = json.loads(path.read_text(encoding="utf-8"))
                            verify_run(path.parent, recovered)
                            interrupted.update(
                                status=recovered["status"],
                                sha256=file_hash(path),
                                total_evaluations=recovered["total_evaluations"],
                                elapsed_ms=recovered["elapsed_ms"],
                                recovered_after_persist=True,
                            )
                        else:
                            interrupted.update(status="failed", interrupted=True, sha256=None, total_evaluations=None)
                        write_json(attempt_file, attempts)
                    progress(
                        {
                            "phase": "running",
                            "case": case,
                            "seed": seed,
                            "strategy": strategy,
                            "completed_runs": len(results),
                        }
                    )
                    if attempts and attempts[-1]["status"] != "failed":
                        saved = attempts[-1]
                        path = (output / saved["result_path"]).resolve()
                        if not path.is_relative_to(output.resolve()) or file_hash(path) != saved["sha256"]:
                            raise WorkbenchError("已完成运行的记录已改变", "SOURCE_CHANGED")
                        result = json.loads(path.read_text(encoding="utf-8"))
                        verify_run(path.parent, result)
                        results.append(
                            {
                                "case": case,
                                "event_id": event,
                                "result_path": saved["result_path"],
                                "attempts": attempts,
                                **result,
                            }
                        )
                        continue
                    folder = base / ("attempt_" + str(len(attempts) + 1))
                    current_attempt = {
                        "status": "running",
                        "result_path": (folder / "result.json").relative_to(output).as_posix(),
                        "sha256": None,
                        "total_evaluations": None,
                        "started_at": utc_now(),
                    }
                    attempts.append(current_attempt)
                    write_json(attempt_file, attempts)
                    attempt_started = perf_counter()
                    try:
                        assembly_started = perf_counter()
                        prepared = build_problem(data, models, request.problem_request(event, seed, bundle))
                        assembly = (perf_counter() - assembly_started) * 1000
                        result = run_portfolio(
                            prepared, strategy, seed, request.total_budget, request.seconds_per_run, folder
                        )
                        result["problem_assembly_ms"] = assembly
                        result["independent_checks"] = verify_run(folder, result)
                        write_json(folder / "result.json", result)
                    except Exception as exc:
                        failure = (
                            json.loads((folder / "failure.json").read_text(encoding="utf-8"))
                            if (folder / "failure.json").exists()
                            else {}
                        )
                        result = {
                            "strategy": strategy,
                            "seed": seed,
                            "status": "failed",
                            "error_code": getattr(exc, "code", type(exc).__name__),
                            "error_message": str(exc)[:300],
                            "total_evaluations": failure.get("total_evaluations"),
                            "evaluations": failure.get("charged_evaluations"),
                            "budget_limit": request.total_budget,
                        }
                        folder.mkdir(parents=True, exist_ok=True)
                        write_json(folder / "failed_result.json", result)
                    path = folder / ("result.json" if result["status"] != "failed" else "failed_result.json")
                    current_attempt.update(
                        status=result["status"],
                        result_path=path.relative_to(output).as_posix(),
                        sha256=file_hash(path),
                        total_evaluations=result["total_evaluations"],
                        elapsed_ms=(perf_counter() - attempt_started) * 1000,
                    )
                    write_json(attempt_file, attempts)
                    results.append(
                        {
                            "case": case,
                            "event_id": event,
                            "result_path": path.relative_to(output).as_posix(),
                            "attempts": attempts,
                            **result,
                        }
                    )
                    write_json(
                        output / "completed_runs.json",
                        [{k: r[k] for k in ("case", "seed", "strategy", "status", "result_path")} for r in results],
                    )
    if source_record(data, models, request)[0] != source:
        raise WorkbenchError("比较期间来源发生变化", "SOURCE_CHANGED")
    summary = paired_summary(results, request.strategies)
    successful = [r for r in results if r["status"] != "failed"]
    prior_failures = sum(any(a["status"] == "failed" for a in r.get("attempts", [])) for r in results)
    result = {
        "schema_version": "portfolio-comparison.g5d.v1",
        "status": "completed" if len(successful) == len(results) and not prior_failures else "completed_with_failures",
        "created_at": utc_now(),
        "runs": len(results),
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
        "runs_with_failed_attempts": prior_failures,
        "cases": len(request.event_ids) if request.mode == "plant" else 1,
        "strategies": list(request.strategies),
        "seeds": list(request.seeds),
        "paired": summary,
        "results": results,
        "total_recorded_evaluations": sum(a["total_evaluations"] or 0 for r in results for a in r.get("attempts", [])),
        "protocol_sha256": file_hash(output / "protocol.json"),
        "run_hashes": {
            a["result_path"]: a["sha256"] for r in results for a in r.get("attempts", []) if a.get("sha256")
        },
        "resources": {
            "elapsed_ms": (perf_counter() - started) * 1000,
            "cpu_seconds": process_time() - cpu_started,
            "peak_process_rss_mb": process_peak_mb(),
            "scope": (
                "comparison source validation, problem assembly, search and aggrega"
                "tion; excludes caller dataset initialization"
            ),
            "llm_calls": 0,
            "tokens": 0,
            "api_cost_cny": 0,
        },
        "runs_with_unknown_failed_cost": sum(
            a.get("total_evaluations") is None for r in results for a in r.get("attempts", [])
        ),
        "execution_authorized": False,
        "method_inventory_increment": 0,
    }
    write_json(output / "comparison.json", safe(result))
    fields = (
        "case",
        "event_id",
        "strategy",
        "seed",
        "status",
        "hv",
        "igd_plus",
        "total_evaluations",
        "elapsed_ms",
        "feasible_rate",
        "verified_front_points",
    )
    pd.DataFrame([{k: r.get(k) for k in fields} for r in results]).to_csv(output / "metrics.csv", index=False)
    report = "# G5d 优化器组合比较\n\n"
    report += (
        "工况 "
        f"{result['cases']}"
        " 个，种子 "
        f"{len(request.seeds)}"
        " 个，策略 "
        f"{len(request.strategies)}"
        " 种。共 "
        f"{len(results)}"
        " 次运行，曾失败或中断 "
        f"{result['runs_with_failed_attempts']}"
        " 次，当前仍失败 "
        f"{result['failed_runs']}"
        " 次。\n"
        "\n"
    )
    report += "| 策略 | 对 NSGA-II 胜 / 平 / 负 | 配对相对 HV 变化中位数 | 耗时比中位数 |\n|---|---|---:|---:|\n"
    for name, item in summary.items():
        gain = "不可计算" if item["median_relative_hv_gain"] is None else f"{item['median_relative_hv_gain']:.4%}"
        ratio = "不可计算" if item["median_elapsed_ratio"] is None else f"{item['median_elapsed_ratio']:.3f}"
        report += f"| {name} | {item['wins']} / {item['ties']} / {item['losses']} | {gain} | {ratio} |\n"
    report += (
        "\n"
        "每一对先在相同问题签名与种子内比较，再汇总相对变化。绝对 HV 不跨工"
        "况排名。数学问题使用解析参考前沿；工厂条件模型没有独立真实前沿，不"
        "报告其 IGD+。\n"
        "\n"
    )
    report += (
        "总预算包含参考、初始点、各臂 pilot、搜索与最终复算。各臂 pilot 使"
        "用独立相同初始种群，完成后才交换同签名前沿；复用次数与实际求值数分"
        "开记录。固定算法也使用相同总预算。\n"
        "\n"
    )
    report += "[完整比较](comparison.json) · [逐运行指标](metrics.csv) · [固定协议](protocol.json)\n\n"
    report += (
        "发生失败后恢复的运行保留所有尝试及成本，并从单次预算的主要配对比较中分开，不能用恢复后的成功覆盖原失败。\n\n"
    )
    report += "本次只比较已有条件模型内的研究候选，未更改设备执行资格或默认优化器。\n"
    (output / "report.md").write_bytes(report.encode("utf-8"))
    return result
