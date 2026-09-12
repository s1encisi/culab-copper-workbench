"""Independent G2a scoring from immutable predictions and mature labels."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from copper_mvp.common import WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_training import source_signature


def read_training(root: Path):
    root = Path(root)
    manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
    for name, key in (("protocol.json", "protocol_sha256"), ("oof_predictions.csv", "oof_sha256")):
        if not (root / name).is_file() or file_hash(root / name) != manifest[key]:
            raise WorkbenchError("比较输入工件校验失败: " + name, "COMPARISON_HASH_MISMATCH")
    if manifest.get("uncertainty_sha256") and file_hash(root / "oof_uncertainty.csv") != manifest["uncertainty_sha256"]:
        raise WorkbenchError("概率预测工件校验失败", "COMPARISON_HASH_MISMATCH")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    return manifest, protocol


def numerical_metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    if not len(error) or not np.isfinite(error).all():
        raise WorkbenchError("共同评价集合为空或含非法值", "COMPARISON_METRICS")
    total = float(np.square(actual - np.mean(actual)).sum())
    return {"n": len(error), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "bias": float(error.mean()), "r2": float(1 - np.square(error).sum() / total) if total > 0 and len(error) > 1 else None}


def paired_week_interval(differences, decisions, settings):
    dates = pd.to_datetime(decisions, utc=True).tz_convert("Asia/Shanghai").tz_localize(None)
    weeks = dates.to_period("W-SUN").astype(str)
    frame = pd.DataFrame({"week": weeks, "difference": differences})
    groups = frame.groupby("week").difference.agg(["sum", "count"])
    if len(groups) < settings["minimum_blocks"]:
        return {"low": None, "high": None, "blocks": len(groups), "reason": "INSUFFICIENT_TIME_BLOCKS"}
    rng = np.random.default_rng(settings["seed"])
    draws = rng.integers(0, len(groups), size=(settings["replicates"], len(groups)))
    means = groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[draws].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "blocks": len(groups), "reason": None}


def evaluate_comparison(data, root: Path):
    root = Path(root)
    if (root / "evaluation.json").exists():
        raise WorkbenchError("该目录已经完成过独立评价", "COMPARISON_EXISTS")
    started = time.perf_counter()
    manifest, protocol = read_training(root)
    if source_signature(data) != manifest["source"] or manifest["source"]["signature"] != protocol["signature"]:
        raise WorkbenchError("评价数据与训练协议来源不同", "COMPARISON_DATA_VERSION")
    methods = protocol["request"]["methods"]
    frame = pd.read_csv(root / "oof_predictions.csv")
    required = {"event_id", "fold_id", "method_id", "decision_at", "fit_cutoff_at", "cu", "as", "status"}
    if not required <= set(frame) or frame.duplicated(["event_id", "method_id"]).any():
        raise WorkbenchError("OOF 预测身份或字段不合法", "COMPARISON_PREDICTIONS")
    expected = {(event, method) for event in data.fold_for for method in methods}
    actual_keys = set(zip(frame.event_id, frame.method_id))
    if actual_keys != expected or len(frame) != len(expected):
        raise WorkbenchError("OOF 预测未保留完整方法×事件集合", "COMPARISON_COVERAGE")
    cutoffs = {f["fold_id"]: source_time(f["fit_cutoff_at"]) for f in protocol["folds"]}
    for row in frame.itertuples(index=False):
        if (row.fold_id != data.fold_for[row.event_id]
            or source_time(row.decision_at) != source_time(data.row(row.event_id).decision_at)
            or source_time(row.fit_cutoff_at) != cutoffs[row.fold_id]
            or source_time(row.fit_cutoff_at) > source_time(row.decision_at)
            or row.status not in ("completed", "failed")):
            raise WorkbenchError("OOF 预测时间或归属错误", "COMPARISON_PREDICTIONS")
    service = DataService(data)
    visible = service.labels.latest(source_time(protocol["evaluation_as_of"]))
    eligible = {e for e in data.fold_for if all((e, t) in visible and visible[(e, t)].quality_eligible
                and visible[(e, t)].value is not None for t in ("cu", "as"))}
    by_method = {m: frame[frame.method_id.eq(m)].set_index("event_id") for m in methods}
    coverage = {}
    common = set(eligible)
    for method, rows in by_method.items():
        good = rows.status.eq("completed") & np.isfinite(rows[["cu", "as"]].to_numpy(float)).all(axis=1)
        ids = set(rows.index[good]) & eligible
        coverage[method] = {"expected_events": len(data.fold_for), "mature_eligible_events": len(eligible),
                            "valid_events": len(ids), "missing_or_failed_events": len(data.fold_for) - len(ids),
                            "coverage": len(ids) / len(data.fold_for)}
        common &= ids
    events = sorted(common, key=lambda e: (data.row(e).decision_at, e))
    metrics = []; fold_metrics = []
    if events:
        decisions = [source_time(data.row(e).decision_at).isoformat() for e in events]
        truth = {target: np.array([visible[(e, target)].value for e in events]) for target in ("cu", "as")}
        for method in methods:
            for target, unit in (("cu", "g/L"), ("as", "mg/L")):
                predicted = by_method[method].loc[events, target].to_numpy(float)
                baseline = by_method["Persistence"].loc[events, target].to_numpy(float)
                actual = truth[target]
                delta_errors = np.abs(predicted - actual) - np.abs(baseline - actual)
                interval = paired_week_interval(delta_errors, decisions, protocol["uncertainty"])
                values = numerical_metrics(actual, predicted)
                base_mae = float(np.abs(baseline - actual).mean())
                metrics.append({"method_id": method, "target": target, "unit": unit, **values,
                                "mae_skill_vs_persistence": 1 - values["mae"] / base_mae if base_mae else None,
                                "paired_mae_difference": float(delta_errors.mean()), "paired_mae_ci95": interval,
                                "negative_predictions": int((predicted < 0).sum())})
                for fold in cutoffs:
                    mask = np.array([data.fold_for[e] == fold for e in events])
                    if mask.any():
                        fold_metrics.append({"method_id": method, "target": target, "unit": unit, "fold_id": fold,
                                             **numerical_metrics(actual[mask], predicted[mask])})
    resources = []
    for method in methods:
        rows = [r for r in manifest["timings"] if r["method_id"] == method]
        cv = [r for r in rows if r["fold_id"] != "DEVELOPMENT"]
        latencies = [value for row in cv for value in row["single_prediction_ms"]]
        resources.append({"method_id": method, "cv_fit_ms": sum(r["fit_ms"] for r in cv),
                          "development_fit_ms": sum(r["fit_ms"] for r in rows if r["fold_id"] == "DEVELOPMENT"),
                          "cv_batch_prediction_ms": sum(r["batch_predict_ms"] for r in cv),
                          "single_predict_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
                          "single_predict_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
                          "timing_samples": len(latencies), "prediction_scope": sorted({r.get("prediction_scope", "single_row_matrix_inference") for r in cv}), "failed_fits_or_predictions": sum(r["status"] == "failed" for r in rows),
                          "fit_warning_count": sum(len(r["warnings"]) for r in rows),
                          "retries": sum(r["retries"] for r in rows), "cache_hits": 0,
                          "llm_calls": 0, "tokens": 0, "api_cost_cny": 0.0})
    if source_signature(data) != manifest["source"]:
        raise WorkbenchError("评价期间数据来源改变", "SOURCE_CHANGED")
    result = safe({
        "schema_version": "model-comparison.g2a.v1", "created_at": utc_now(),
        "status": "completed" if events and manifest["status"] == "completed" and len(events) == len(data.fold_for) else "partial_failure",
        "evaluation_mode": "historical_replay", "methods": methods, "evaluation_as_of": protocol["evaluation_as_of"],
        "common_events": len(events), "common_event_hash": digest(events), "coverage": coverage,
        "metrics": metrics, "fold_metrics": fold_metrics, "resources": resources,
        "uncertainty": protocol["uncertainty"], "runtime": manifest["runtime"],
        "training_manifest_sha256": file_hash(root / "training_manifest.json"),
        "protocol_sha256": manifest["protocol_sha256"], "oof_sha256": manifest["oof_sha256"],
        "evaluator_sha256": file_hash(Path(__file__)), "evaluator_elapsed_ms": (time.perf_counter() - started) * 1000,
        "automatic_promotion": False, "optimization_proxy_approval": False,
        "best_observed_mae": {t: min((r for r in metrics if r["target"] == t), key=lambda r: r["mae"])["method_id"]
                              for t in ("cu", "as")} if metrics else {},
        "note": "固定参数的开发期 OOF 比较；周块重采样区间依赖时间块假设，不是独立外评或因果证据。",
    })
    if manifest.get("uncertainty_sha256"):
        from copper_mvp.classical_evaluation import uncertainty_metrics
        result["uncertainty_metrics"] = uncertainty_metrics(root / "oof_uncertainty.csv", visible, events)
        result["uncertainty_sha256"] = manifest["uncertainty_sha256"]
    write_json(root / "evaluation.json", result)
    csv_rows = [{**{k: v for k, v in row.items() if k != "paired_mae_ci95"},
                 "paired_mae_ci95_low": row["paired_mae_ci95"]["low"],
                 "paired_mae_ci95_high": row["paired_mae_ci95"]["high"]} for row in metrics]
    pd.DataFrame(csv_rows).to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(root / "fold_metrics.csv", index=False)
    report = "# G2a 模型比较报告\n\n"
    report += f"状态：{result['status']}。方法 {len(methods)} 种，共同 OOF 事件 {len(events)} 个。所有方法使用相同事件集合，原输出单位保留。\n\n"
    report += "| 方法 | 目标 | MAE | RMSE | R² | 偏差 | 相对 Persistence 的 MAE 差 |\n|---|---|---:|---:|---:|---:|---:|\n"
    for row in metrics:
        r2 = f"{row['r2']:.4f}" if row["r2"] is not None else "不可计算"
        report += f"| {row['method_id']} | {row['target'].upper()} ({row['unit']}) | {row['mae']:.6g} | {row['rmse']:.6g} | {r2} | {row['bias']:.6g} | {row['paired_mae_difference']:.6g} |\n"
    report += "\nMAE 差为本方法减 Persistence，负值表示本次共同事件上的误差更低。完整 CSV/JSON 保留周块重采样区间和各折结果。\n\n"
    report += "| 方法 | 五折拟合(ms) | 单事件推理 p50(ms) | p95(ms) | 计时样本 | 拟合警告 | 失败 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for row in resources:
        p50 = f"{row['single_predict_p50_ms']:.4f}" if row["single_predict_p50_ms"] is not None else "缺失"
        p95 = f"{row['single_predict_p95_ms']:.4f}" if row["single_predict_p95_ms"] is not None else "缺失"
        report += f"| {row['method_id']} | {row['cv_fit_ms']:.2f} | {p50} | {p95} | {row['timing_samples']} | {row['fit_warning_count']} | {row['failed_fits_or_predictions']} |\n"
    report += "\n推理计时不含排队、磁盘加载或 HTTP 开销；事件序列方法还重建截至该事件的因果历史。JSON 中分别记录矩阵推理与上下文重建范围，不能直接混比。\n\n"
    report += "[完整评价](evaluation.json) · [指标 CSV](metrics.csv) · [各折指标](fold_metrics.csv) · [预先固定协议](protocol.json) · [训练和工件清单](training_manifest.json)\n\n"
    report += "预测排名和优化响应代理资格分开登记。本次比较不改变旧版默认模型、不自动批准新代理，也不使用 2026 外评。方法和参数均在查看本轮指标前固定。\n"
    (root / "report.md").write_text(report, encoding="utf-8")
    return result
