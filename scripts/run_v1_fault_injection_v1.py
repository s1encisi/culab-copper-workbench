"""生成 P3 开发期离线故障注入、安全消融与延迟基准产物。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_mas.evaluation.fault_injection import (  # noqa: E402
    DIRECT_SYSTEM,
    GUARDED_SYSTEM,
    SINGLE_LLM_SYSTEM,
    assert_benchmark_expectations,
    execute_fault_benchmark_v1,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: pd.Series, percentile: float) -> float:
    return float(np.percentile(values.to_numpy(dtype=float), percentile, method="linear"))


def _scenario_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = results.groupby(["scenario_id", "场景名称", "system"], sort=True)
    for (scenario_id, name, system), group in grouped:
        rows.append(
            {
                "scenario_id": scenario_id,
                "场景名称": name,
                "system": system,
                "repeats": int(len(group)),
                "is_fault": bool(group["is_fault"].iloc[0]),
                "prediction_emitted_count": int(group["prediction_emitted"].sum()),
                "unsafe_prediction_emitted_count": int(
                    group["unsafe_prediction_emitted"].sum()
                ),
                "failure_detected_count": int(group["failure_detected"].sum()),
                "abstained_or_blocked_count": int(
                    group["abstained_or_blocked"].sum()
                ),
                "unsafe_prediction_rate": float(
                    group["unsafe_prediction_emitted"].mean()
                ),
                "failure_detection_rate": float(group["failure_detected"].mean()),
                "abstain_or_block_rate": float(
                    group["abstained_or_blocked"].mean()
                ),
                "latency_p50_ms": _percentile(group["latency_ms"], 50),
                "latency_p95_ms": _percentile(group["latency_ms"], 95),
                "reason_codes": "|".join(sorted(set(group["reason_code"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def _system_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system, group in results.groupby("system", sort=True):
        faults = group[group["is_fault"]]
        clean = group[~group["is_fault"]]
        rows.append(
            {
                "system": system,
                "status": "EXECUTED_OFFLINE",
                "evaluated_runs": int(len(group)),
                "fault_runs": int(len(faults)),
                "unsafe_prediction_rate_on_faults": float(
                    faults["unsafe_prediction_emitted"].mean()
                ),
                "failure_detection_rate_on_faults": float(
                    faults["failure_detected"].mean()
                ),
                "abstain_or_block_rate_on_faults": float(
                    faults["abstained_or_blocked"].mean()
                ),
                "clean_prediction_pass_rate": float(clean["prediction_emitted"].mean()),
                "latency_p50_ms_all": _percentile(group["latency_ms"], 50),
                "latency_p95_ms_all": _percentile(group["latency_ms"], 95),
                "llm_calls": int(group["llm_call_count"].sum()),
                "network_calls": int(group["network_call_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _status_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "system": DIRECT_SYSTEM,
                "status": "EXECUTED_OFFLINE",
                "说明": "无合同、准入、完整性和结构化输出校验的固定预测消融基线。",
            },
            {
                "system": GUARDED_SYSTEM,
                "status": "EXECUTED_OFFLINE",
                "说明": "本地确定性 A1→A2→A4→A5 图及外层 fail-closed 防护。",
            },
            {
                "system": SINGLE_LLM_SYSTEM,
                "status": "PENDING_AUTHORIZATION",
                "说明": "未获真实调用授权、地区确认与预算上限前不运行，也不伪造结果。",
            },
        ]
    )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_report(
    *, repeats: int, scenarios: pd.DataFrame, system: pd.DataFrame
) -> str:
    direct = system.set_index("system").loc[DIRECT_SYSTEM]
    guarded = system.set_index("system").loc[GUARDED_SYSTEM]
    direct_p50 = float(direct["latency_p50_ms_all"])
    guarded_p50 = float(guarded["latency_p50_ms_all"])
    overhead = guarded_p50 - direct_p50
    ratio = guarded_p50 / direct_p50 if direct_p50 > 0 else float("nan")

    scenario_lines = []
    for row in scenarios.itertuples(index=False):
        fault_text = "故障" if bool(getattr(row, "是否故障")) else "正常对照"
        scenario_lines.append(
            f"| {row.scenario_id} | {getattr(row, '场景名称')} | {fault_text} | "
            f"{getattr(row, '预期防护原因码')} |"
        )

    return f"""# P3 离线故障注入与安全消融报告 V1

## 一、结论

本轮只回答一个问题：多智能体工作流相对“直接固定预测”增加的价值，能否落实为可测的准入、审计和拒绝能力。结果表明，在 {len(scenarios) - 1} 类故障、每类 {repeats} 次重复的离线实验中：

- `direct_fixed_predictor` 对故障输入的危险预测输出率为 {_pct(float(direct['unsafe_prediction_rate_on_faults']))}，故障发现率为 {_pct(float(direct['failure_detection_rate_on_faults']))}；这是有意移除安全层的消融基线。
- `guarded_multi_agent_graph` 对故障输入的危险预测输出率为 {_pct(float(guarded['unsafe_prediction_rate_on_faults']))}，故障发现率和拒绝/弃权率均为 {_pct(float(guarded['failure_detection_rate_on_faults']))}。
- 正常对照在两条路径上的预测通过率均为 {_pct(float(guarded['clean_prediction_pass_rate']))}，未出现“所有输入一律拒绝”的退化做法。
- 全场景离线延迟中位数由 {direct_p50:.6f} ms 增至 {guarded_p50:.6f} ms，绝对增加 {overhead:.6f} ms，约为 {ratio:.2f} 倍。该数值只反映当前电脑、Python 进程和微秒级本地校验开销，不能外推为现场部署延迟。

因此，这一实验支持的论文表述是：多智能体工作流通过合同化分工和独立审计减少不安全输出；它不支持“增加智能体就能提高 Cu/As 预测精度”的主张。数值精度仍应由同一冻结 A4 模型、同一数据与同一评价集单独比较。

## 二、实验边界

- 数据边界：只使用合成的 2025 开发期格式样本，不读取 2026 外部时序留出集，也不读取任何 `outcome_ledger`。
- 模型边界：数值预测由本地固定函数产生；本实验不训练、不选择模型，也不计算 Cu/As 的 MAE、RMSE 或 R²。
- LLM 边界：没有发起 Kimi 或 DeepSeek 调用；`live_calls=false` 场景在网络连接前被拒绝。`single_llm_agent` 仅登记为 `PENDING_AUTHORIZATION`，没有伪造对照结果。
- 重复设计：11 个场景分别重复 {repeats} 次，另有 3 次不计入结果的预热；同一重复内交替两个系统的先后次序，并报告 p50/p95。
- 安全指标：`unsafe_prediction_emitted` 表示在已知故障尚未被发现时仍产生正式预测；`failure_detected` 表示防护层给出稳定原因码；`abstained_or_blocked` 同时覆盖数据不足时的弃权和合同违规时的阻断。

## 三、场景设计

| 场景 ID | 场景 | 类型 | 防护图预期原因码 |
|---|---|---|---|
{chr(10).join(scenario_lines)}

其中，三四段整组缺失被定义为 `ABSTAINED`，因为它属于信息不足；其余合同违规、模型篡改和非法输出被定义为 `BLOCKED`。正常对照必须通过完整 A1→A2→A4→A5 路径。

## 四、比较结果

| 系统 | 故障危险输出率 | 故障发现率 | 故障拒绝/弃权率 | 正常通过率 | p50 延迟(ms) | p95 延迟(ms) |
|---|---:|---:|---:|---:|---:|---:|
| direct_fixed_predictor | {_pct(float(direct['unsafe_prediction_rate_on_faults']))} | {_pct(float(direct['failure_detection_rate_on_faults']))} | {_pct(float(direct['abstain_or_block_rate_on_faults']))} | {_pct(float(direct['clean_prediction_pass_rate']))} | {float(direct['latency_p50_ms_all']):.6f} | {float(direct['latency_p95_ms_all']):.6f} |
| guarded_multi_agent_graph | {_pct(float(guarded['unsafe_prediction_rate_on_faults']))} | {_pct(float(guarded['failure_detection_rate_on_faults']))} | {_pct(float(guarded['abstain_or_block_rate_on_faults']))} | {_pct(float(guarded['clean_prediction_pass_rate']))} | {float(guarded['latency_p50_ms_all']):.6f} | {float(guarded['latency_p95_ms_all']):.6f} |

逐场景结果保存在 `scenario_summary_v1.csv`，每次重复的原始结果保存在 `per_run_results_v1.csv`。模型哈希篡改场景校验的是冻结标识与 SHA-256 一致性；LLM 两个场景分别校验离线开关和本地 Pydantic 输出合同。

## 五、论文中可以与不可以写的内容

可以写：

1. 防护图在预定义故障集上实现了可追踪的 fail-closed 行为，并输出稳定原因码。
2. 相比无防护固定预测器，防护图消除了本次故障集中的危险预测输出，代价是可量化的本地校验延迟。
3. 多智能体的必要性来自职责隔离：A1 管准入，A2 管工况描述，A4 只执行冻结数值模型，A5 独立审计；LLM 不是数值预测器，也不能覆盖安全判断。

不可以写：

1. 不得把本实验解释为多智能体提升了 Cu/As 预测精度。
2. 不得把合成故障的 100% 检出率外推为未知现场故障的 100% 检出率。
3. 不得声称完成了单 LLM 智能体比较；该对照尚未获真实调用授权。
4. 不得把微秒/毫秒级本地离线延迟等同于未来 API、网络或现场系统延迟。

## 六、后续建议

下一步可在不改变 2026 留出规则的前提下，将人工抽核发现的真实数据问题逐项加入场景库，并冻结场景版本。待密钥轮换、Kimi 地区和月度预算得到确认后，再单独运行 `single_llm_agent`；该结果必须记录模型、提示词版本、调用量、token、成本、失败率和 p50/p95，不能回填或虚构。
"""


def run_benchmark(output_dir: Path, *, repeats: int = 100) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_objects, result_objects = execute_fault_benchmark_v1(repeats=repeats)
    assert_benchmark_expectations(scenario_objects, result_objects)

    scenarios = pd.DataFrame([item.definition_row() for item in scenario_objects])
    results = pd.DataFrame([item.as_row() for item in result_objects])
    scenario_summary = _scenario_summary(results)
    system_summary = _system_summary(results)
    status = _status_table()

    paths = {
        "scenario_csv": output_dir / "scenario_definitions_v1.csv",
        "scenario_json": output_dir / "scenario_definitions_v1.json",
        "per_run": output_dir / "per_run_results_v1.csv",
        "scenario_summary": output_dir / "scenario_summary_v1.csv",
        "system_summary": output_dir / "system_summary_v1.csv",
        "system_status": output_dir / "system_status_v1.csv",
        "report": output_dir / "P3离线故障注入与安全消融报告_V1.md",
    }
    scenarios.to_csv(paths["scenario_csv"], index=False, encoding="utf-8-sig")
    paths["scenario_json"].write_text(
        json.dumps(
            scenarios.to_dict(orient="records"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    results.to_csv(
        paths["per_run"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.9f",
    )
    scenario_summary.to_csv(
        paths["scenario_summary"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.9f",
    )
    system_summary.to_csv(
        paths["system_summary"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.9f",
    )
    status.to_csv(paths["system_status"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        _render_report(repeats=repeats, scenarios=scenarios, system=system_summary),
        encoding="utf-8",
    )

    hash_rows = [
        {"artifact": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in paths.values()
    ]
    hash_path = output_dir / "artifact_hashes_v1.csv"
    pd.DataFrame(hash_rows).to_csv(hash_path, index=False, encoding="utf-8-sig")

    fault_rows = results[results["is_fault"]]
    guarded_faults = fault_rows[fault_rows["system"] == GUARDED_SYSTEM]
    direct_faults = fault_rows[fault_rows["system"] == DIRECT_SYSTEM]
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_id": "P3_FAULT_INJECTION_SAFETY_ABLATION_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "DEVELOPMENT_OFFLINE_SAFETY_ONLY",
        "scenario_count": int(len(scenarios)),
        "fault_scenario_count": int(scenarios["是否故障"].sum()),
        "repeats_per_scenario": repeats,
        "warmup_repeats_not_reported": 3,
        "measured_run_count": int(len(results)),
        "executed_systems": [DIRECT_SYSTEM, GUARDED_SYSTEM],
        "single_llm_agent_status": "PENDING_AUTHORIZATION",
        "external_2026_read": False,
        "outcome_ledger_read": False,
        "llm_calls_made": 0,
        "network_calls_made": 0,
        "target_values_read": 0,
        "accuracy_claim_made": False,
        "guarded_fault_detection_rate": float(guarded_faults["failure_detected"].mean()),
        "guarded_unsafe_prediction_rate": float(
            guarded_faults["unsafe_prediction_emitted"].mean()
        ),
        "direct_unsafe_prediction_rate": float(
            direct_faults["unsafe_prediction_emitted"].mean()
        ),
        "all_expectations_passed": True,
        "hash_inventory": hash_path.name,
        "artifact_hashes": {row["artifact"]: row["sha256"] for row in hash_rows},
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/v1_fault_injection_v1",
    )
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    manifest = run_benchmark(args.output_dir, repeats=args.repeats)
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "scenario_count": manifest["scenario_count"],
                "fault_scenario_count": manifest["fault_scenario_count"],
                "measured_run_count": manifest["measured_run_count"],
                "guarded_fault_detection_rate": manifest[
                    "guarded_fault_detection_rate"
                ],
                "guarded_unsafe_prediction_rate": manifest[
                    "guarded_unsafe_prediction_rate"
                ],
                "external_2026_read": manifest["external_2026_read"],
                "llm_calls_made": manifest["llm_calls_made"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
