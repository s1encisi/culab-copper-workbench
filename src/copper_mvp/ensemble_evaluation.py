"""Independent ensemble scoring over a common temporal cohort and every seed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from copper_mvp.common import WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_evaluation import numerical_metrics
from copper_mvp.model_training import source_signature


def hierarchical_interval(differences, decisions, seed=20260911, replicates=1000):
    differences = np.asarray(differences, float)
    dates = pd.to_datetime(decisions, utc=True).tz_convert("Asia/Shanghai").tz_localize(None)
    blocks = dates.to_period("W-SUN").astype(str)
    unique = sorted(set(blocks))
    if len(unique) < 20:
        return {
            "low": None,
            "high": None,
            "blocks": len(unique),
            "seeds": len(differences),
            "status": "INSUFFICIENT_BLOCKS",
        }
    sums = np.array([[values[blocks == block].sum() for block in unique] for values in differences])
    counts = np.array([(blocks == block).sum() for block in unique])
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, len(differences), (replicates, len(differences)))
    block_draws = rng.integers(0, len(unique), (replicates, len(unique)))
    sample_sums = sums[seed_draws[:, :, None], block_draws[:, None, :]].sum(axis=(1, 2))
    means = sample_sums / (counts[block_draws].sum(axis=1) * len(differences))
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "blocks": len(unique),
        "seeds": len(differences),
        "status": "COMPUTED",
        "method": "seed_and_calendar_week_block_bootstrap",
        "replicates": replicates,
        "assumption": "seed and between-week resampling conditional on these fixed folds and model specifications",
    }


def error_correlation(predictions, actual, names):
    errors = np.asarray(predictions) - np.asarray(actual)[:, None]
    centered = errors - errors.mean(axis=0)
    norms = np.sqrt(np.square(centered).sum(axis=0))
    result = []
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            row.append(
                float(centered[:, i] @ centered[:, j] / (norms[i] * norms[j])) if norms[i] * norms[j] > 0 else None
            )
        result.append(row)
    return {"methods": list(names), "correlations": result}


def verify_fold(root, manifest, book, expected_fold):
    for name, key in (
        ("predictions.csv", "predictions_sha256"),
        ("weight_decisions.jsonl", "decisions_sha256"),
        ("inner_oof.csv", "inner_oof_sha256"),
        ("inner_plan.json", "inner_plan_sha256"),
        ("selection.json", "selection_sha256"),
        ("bagging_selection.json", "bagging_selection_sha256"),
    ):
        if file_hash(root / name) != manifest[key]:
            raise WorkbenchError("集成折工件哈希不符: " + name, "ENSEMBLE_HASH_MISMATCH")
    for artifact in manifest["artifacts"]:
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != artifact["sha256"]:
            raise WorkbenchError("集成模型工件哈希不符", "ENSEMBLE_HASH_MISMATCH")
    plan = json.loads((root / "inner_plan.json").read_text(encoding="utf-8"))
    if set(plan["outer_train_ids"]) != set(expected_fold["train_event_ids"]) or set(
        plan["outer_validation_ids"]
    ) != set(expected_fold["validation_event_ids"]):
        raise WorkbenchError("外层折事件与固定协议不一致", "ENSEMBLE_OUTER_SPLIT")
    seen = []
    for part in plan["inner_folds"]:
        train, valid = part["train_event_ids"], part["validation_event_ids"]
        cutoff = source_time(part["fit_cutoff_at"])
        if (
            set(train) & set(valid)
            or set(train) - set(plan["outer_train_ids"])
            or set(valid) - set(plan["outer_train_ids"])
            or any(book.decision(e) >= cutoff for e in train)
            or any(book.decision(e) < cutoff for e in valid)
        ):
            raise WorkbenchError("内层 OOF 时间或事件边界失效", "ENSEMBLE_INNER_SPLIT")
        actual = book.evidence(train, cutoff)
        if any(actual[k] != part[k] for k in actual):
            raise WorkbenchError("内层训练标签版本与记录不一致", "ENSEMBLE_LABEL_VERSION")
        seen.extend(valid)
    oof = pd.read_csv(root / "inner_oof.csv")
    if len(set(seen)) != len(seen) or set(oof.event_id) != set(seen) or len(oof) != len(seen):
        raise WorkbenchError("内层 OOF 包含重叠、缺失或暖启动填充行", "ENSEMBLE_OOF_COVERAGE")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    meta = selection["meta_selection_train"]
    eligible = book.mature(
        [e for e in oof.event_id if book.decision(e) < source_time(meta["fit_cutoff_at"])], meta["fit_cutoff_at"]
    )
    if book.evidence(eligible, meta["fit_cutoff_at"]) != meta:
        raise WorkbenchError("二层模型选择使用了错误的标签版本", "ENSEMBLE_META_SPLIT")
    return True


def evaluate_ensembles(data, root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "replay_manifest.json").read_text(encoding="utf-8"))
    for name, key in (
        ("protocol.json", "protocol_sha256"),
        ("predictions.csv", "predictions_sha256"),
        ("decisions.jsonl", "decisions_sha256"),
    ):
        if file_hash(root / name) != manifest[key]:
            raise WorkbenchError("集成比较工件哈希不符", "ENSEMBLE_HASH_MISMATCH")
    if source_signature(data) != protocol["source"]:
        raise WorkbenchError("集成评价的数据来源改变", "SOURCE_CHANGED")
    frame = pd.read_csv(root / "predictions.csv")
    frame["seed_key"] = frame.seed.fillna(-1).astype(int)
    expected = {(e, m, s) for e in data.fold_for for m, seeds in protocol["expected_methods"].items() for s in seeds}
    if (
        frame.duplicated(["event_id", "method_id", "seed_key"]).any()
        or set(zip(frame.event_id, frame.method_id, frame.seed_key, strict=True)) != expected
    ):
        raise WorkbenchError("比较没有保留完整方法、种子和事件集合", "ENSEMBLE_COVERAGE")
    for row in frame.itertuples():
        if row.fold_id != data.fold_for[row.event_id] or source_time(row.fit_cutoff_at) > source_time(row.decision_at):
            raise WorkbenchError("外层预测时间或归属错误", "ENSEMBLE_PREDICTION_TIME")
    labels = DataService(data).labels.latest(protocol["evaluation_as_of"])
    eligible = {
        e
        for e in data.fold_for
        if all(
            (e, t) in labels and labels[(e, t)].quality_eligible and labels[(e, t)].value is not None
            for t in ("cu", "as")
        )
    }
    common = set(eligible)
    groups = {}
    coverage = {}
    for (method, seed), part in frame.groupby(["method_id", "seed_key"]):
        rows = part.set_index("event_id")
        good = rows.status.eq("completed") & np.isfinite(rows[["cu", "as"]].to_numpy(float)).all(axis=1)
        valid = set(rows.index[good])
        common &= valid
        groups[(method, int(seed))] = rows
        coverage[f"{method}:{seed}"] = {
            "expected_events": len(data.fold_for),
            "valid_predictions": len(valid),
            "failed_or_missing": len(data.fold_for) - len(valid),
        }
    events = sorted(common, key=lambda e: (data.row(e).decision_at, e))
    if not events:
        raise WorkbenchError("集成比较没有共同可评分事件", "ENSEMBLE_NO_EVALUATION")
    times = [source_time(data.row(e).decision_at).isoformat() for e in events]
    metric_rows = []
    seed_rows = []
    residuals = {}
    baseline = groups[("Persistence", -1)].loc[events, ["cu", "as"]].to_numpy()
    for target_index, (target, unit) in enumerate((("cu", "g/L"), ("as", "mg/L"))):
        actual = np.array([labels[(e, target)].value for e in events])
        base_error = np.abs(baseline[:, target_index] - actual)
        names = protocol["base_methods"]
        residuals[target] = error_correlation(
            np.column_stack([groups[(m, -1)].loc[events, target].to_numpy() for m in names]), actual, names
        )
        for method, seeds in protocol["expected_methods"].items():
            values = np.stack([groups[(method, seed)].loc[events, target].to_numpy() for seed in seeds])
            errors = np.abs(values - actual)
            seed_mae = errors.mean(axis=1)
            for seed, predicted in zip(seeds, values, strict=True):
                seed_rows.append(
                    {
                        "method_id": method,
                        "seed": None if seed == -1 else seed,
                        "target": target,
                        "unit": unit,
                        **numerical_metrics(actual, predicted),
                    }
                )
            by_mode = {}
            by_fold = {}
            for mapping, attribute in (
                (by_mode, lambda e: data.mode_cards[e]["mode_code"]),
                (by_fold, lambda e: data.fold_for[e]),
            ):
                for group in sorted({attribute(e) for e in events}):
                    mask = np.array([attribute(e) == group for e in events])
                    mapping[group] = {
                        "n": int(mask.sum()),
                        "mae_mean_over_seeds": float(errors[:, mask].mean()),
                        "bias_mean_over_seeds": float((values[:, mask] - actual[mask]).mean()),
                    }
            metric_rows.append(
                {
                    "method_id": method,
                    "target": target,
                    "unit": unit,
                    "n": len(events),
                    "seed_count": len(seeds),
                    "mae": float(seed_mae.mean()),
                    "seed_mae_sd": float(seed_mae.std(ddof=1)) if len(seeds) > 1 else None,
                    "rmse": float(np.sqrt(np.square(values - actual).mean(axis=1)).mean()),
                    "bias": float((values - actual).mean()),
                    "paired_mae_difference": float((errors - base_error).mean()),
                    "paired_ci95": hierarchical_interval(errors - base_error, times),
                    "mode_metrics": by_mode,
                    "fold_metrics": by_fold,
                    "negative_predictions": int((values < 0).sum()),
                    "aggregation": "mean score of individual fitted seeds, not the score of an average across seeds",
                }
            )
    resources = {}
    for fold in manifest["folds"]:
        for artifact in fold["artifacts"]:
            method = artifact.get("method", artifact["name"])
            row = resources.setdefault(
                method, {"samples": [], "scope": artifact["timing_scope"], "fold_seed_artifacts": 0}
            )
            row["samples"].extend(artifact["latency_samples_ms"])
            row["fold_seed_artifacts"] += 1
    resources = {
        m: {
            "p50_ms": float(np.percentile(r["samples"], 50)),
            "p95_ms": float(np.percentile(r["samples"], 95)),
            "samples": len(r["samples"]),
            "scope": r["scope"],
            "fold_seed_artifacts": r["fold_seed_artifacts"],
        }
        for m, r in resources.items()
    }
    result = safe(
        {
            "schema_version": "ensemble-evaluation.g5b.v1",
            "created_at": utc_now(),
            "status": "completed",
            "evaluation_mode": "historical_replay",
            "common_events": len(events),
            "common_event_hash": digest(events),
            "metrics": metric_rows,
            "seed_metrics": seed_rows,
            "coverage": coverage,
            "residual_correlations": residuals,
            "resources": resources,
            "training_resources": manifest["training_resources"],
            "replay_manifest_sha256": file_hash(root / "replay_manifest.json"),
            "protocol_sha256": manifest["protocol_sha256"],
            "live_model_change": False,
            "optimization_proxy_approval": False,
            "causal_control": False,
            "method_inventory_increment": 0,
            "evaluator_sha256": file_hash(Path(__file__)),
        }
    )
    write_json(root / "evaluation.json", result)
    flat = [
        {k: v for k, v in row.items() if k not in ("paired_ci95", "mode_metrics", "fold_metrics")}
        for row in metric_rows
    ]
    pd.DataFrame(flat).to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(root / "seed_metrics.csv", index=False)
    text = "# G5b 嵌套 OOF 集成比较\n\n"
    text += (
        "共同评价 "
        f"{len(events)}"
        " 个原外层 OOF 事件。Bagging 使用 "
        f"{len(protocol['request']['seeds'])}"
        " 个固定种子；二层模型和融合超参数仅在各外层训练段内选择。\n"
        "\n"
    )
    text += "| 方法 / 对照 | 目标 | 平均 MAE | 种子间 SD | 相对 Persistence 的 MAE 差 |\n|---|---|---:|---:|---:|\n"
    for row in metric_rows:
        sd = "—" if row["seed_mae_sd"] is None else f"{row['seed_mae_sd']:.6g}"
        text += (
            "| "
            f"{row['method_id']}"
            " | "
            f"{row['target']}"
            " ("
            f"{row['unit']}"
            ") | "
            f"{row['mae']:.6g}"
            " | "
            f"{sd}"
            " | "
            f"{row['paired_mae_difference']:.6g}"
            " |\n"
        )
    text += (
        "\n"
        "分数先逐个已拟合种子计算，再求均值；没有把多个种子的平均预测当成单"
        "次模型结果。区间联合重采样种子与日历周块。Bagging_d1/d7 是同一方法"
        "的块长度对照，不计入新增方法数。\n"
        "\n"
    )
    text += (
        "MeanEnsemble 与 Stacking/OOFConvex 共享相同的全训练段基模型；Dynam"
        "icConvex 的更新只使用已可得标签。Blending 保留较早训练的基模型，并"
        "使用独立尾段拟合权重。\n"
        "\n"
    )
    text += (
        "训练账本记录实际所有拟合，多个对照共享的基模型只计一次。原 G2a 的 "
        "ElasticNet、Huber、PLS 预测作为历史参考保留，本次没有重训这三种方"
        "法。动态权重更新时间与静态模型完整推理时间分别列示。\n"
        "\n"
    )
    text += (
        "[完整评价](evaluation.json) · [逐种子指标](seed_metrics.csv) · [固"
        "定协议](protocol.json) · [训练与工件账本](replay_manifest.json) · "
        "[外层预测](predictions.csv)\n"
        "\n"
    )
    text += "预测默认值、优化代理和设备命令均未由本次比较改写；是否发布由后续影子与发布流程决定。\n"
    (root / "report.md").write_bytes(text.encode("utf-8"))
    return result
