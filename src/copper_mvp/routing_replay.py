"""Predeclared temporal replay of fixed predictors and deterministic shadow routes."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from copper_mvp.common import WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_evaluation import read_training, numerical_metrics
from copper_mvp.model_training import source_signature, load_registered_model
from copper_mvp.prediction_router import PolicyRouter
from copper_mvp.routing_metrics import MatureMetrics, RoutingPolicy, block_interval


def prepare_replay(data, comparison_root, output):
    manifest, source_protocol = read_training(comparison_root)
    if manifest["status"] != "completed" or source_signature(data) != manifest["source"]:
        raise WorkbenchError("请选择与当前数据一致的已完成模型比较", "ROUTING_SOURCE")
    ledger = DataService(data).labels
    first_fold = source_protocol["folds"][0]
    train_ids = data.training_ids(first_fold["fold_id"])
    cutoff = source_time(first_fold["fit_cutoff_at"])
    visible = ledger.latest(cutoff)
    y = np.array([[visible[(e, t)].value for t in ("cu", "as")] for e in train_ids])
    scales = {t: float(np.percentile(y[:, j], 75)-np.percentile(y[:, j], 25)) for j, t in enumerate(("cu", "as"))}
    if any(not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise WorkbenchError("首个训练段的目标 IQR 无法用于归一化", "ROUTING_SCALE")
    train_gaps = [(visible[(e, "cu")].available_at-visible[(e, "cu")].decision_at).total_seconds()/3600 for e in train_ids]
    boundaries = np.quantile(train_gaps, [1/3, 2/3]).tolist()
    profiles = {}
    with threadpool_limits(limits=1):
        X = data.X.loc[train_ids].to_numpy(float)
        positions = np.unique(np.linspace(0, len(X)-1, min(32, len(X))).astype(int))
        for method in source_protocol["request"]["methods"]:
            model, artifact = load_registered_model(comparison_root, manifest, source_protocol, method, first_fold["fold_id"])
            model.predict(X[:1])
            samples = []
            for position in positions:
                start = time.perf_counter()
                model.predict(X[position:position+1])
                samples.append((time.perf_counter()-start)*1000)
            profiles[method] = {"p50_ms": float(np.percentile(samples,50)), "p95_ms": float(np.percentile(samples,95)),
                "samples": len(samples), "artifact_sha256": artifact["sha256"], "scope": "first_outer_training_rows_warm_two_target_inference",
                "virtual_available_at": cutoff.isoformat(), "api_cost_cny": 0}
    attributes = {}
    initial = {(r.event_id,r.target):r for r in ledger.records if r.revision == 1}
    for event in data.fold_for:
        label = initial[(event, "cu")]
        gap = (label.available_at-label.decision_at).total_seconds()/3600
        attributes[event] = {"decision_at": source_time(data.row(event).decision_at).isoformat(),
            "mode": data.mode_cards[event]["mode_code"],
            "quality": "complete_features" if np.isfinite(data.X.loc[event].to_numpy(float)).all() else "missing_features",
            "interval_group": ("short", "medium", "long")[int(gap > boundaries[0])+int(gap > boundaries[1])],
            "target_available_interval_hours": gap}
    protocol = {"schema_version": "prediction-routing-study.g5a.v1", "created_at": utc_now(),
        "evaluation_mode": "historical_replay", "online_prediction_claim": False, "source": manifest["source"],
        "source_protocol_sha256": file_hash(comparison_root/"protocol.json"), "source_oof_sha256": manifest["oof_sha256"],
        "source_training_manifest_sha256": file_hash(comparison_root/"training_manifest.json"),
        "source_methods": source_protocol["request"]["methods"], "evaluation_as_of": source_protocol["evaluation_as_of"],
        "calibration": {"fit_cutoff_at": cutoff.isoformat(), "train_count": len(train_ids), "train_event_hash": digest(train_ids),
            "label_revision_hash": digest([visible[(e,t)].record_id for e in train_ids for t in ("cu","as")]),
            "target_iqr": scales, "interval_boundaries_hours": boundaries, "resource_profiles": profiles,
            "interval_group_use": "retrospective mature-label stratification only; never a current-event routing input"},
        "strategies": [
            {"id": "rule_global", "by_mode": False, "policy": RoutingPolicy().model_dump()},
            {"id": "rule_mode", "by_mode": True, "policy": RoutingPolicy().model_dump()},
            {"id": "rule_mode_7day_blocks", "by_mode": True, "policy": RoutingPolicy(block_days=7).model_dump()}],
        "parameter_policy": "fixed_predeclared_design_defaults_not_tuned_on_replay_results",
        "confirmation_windows": "new sets of at least 60 initially mature events; daily refreshes do not count as extra confirmations",
        "pool_scope": "benchmarked source methods eligible for offline shadow study only",
        "live_champion": "unchanged", "optimization_proxy_approval": False, "causal_control": False,
        "external_2026_read": False, "code_hashes": {n:file_hash(Path(__file__).with_name(n))
            for n in ("routing_metrics.py","prediction_router.py","routing_replay.py")}}
    write_json(output/"protocol.json", protocol)
    predictions = pd.read_csv(comparison_root/"oof_predictions.csv")
    return protocol, ledger, attributes, predictions


def run_replay(data, comparison_root: Path, output: Path, progress=lambda value: None, max_wall_seconds=600):
    output, comparison_root = Path(output), Path(comparison_root)
    if (output/"protocol.json").exists():
        raise WorkbenchError("该回放目录已有协议，请使用新目录", "ROUTING_EXISTS")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    progress({"phase":"calibration"})
    protocol, ledger, attributes, forecasts = prepare_replay(data, comparison_root, output)
    events = sorted(attributes, key=lambda e:(attributes[e]["decision_at"],e))
    records, decisions, resources = [], [], []
    for method in protocol["source_methods"]:
        for row in forecasts[forecasts.method_id.eq(method)].to_dict("records"):
            records.append({"strategy":"fixed_"+method, "event_id":row["event_id"], "decision_at":row["decision_at"],
                "cu":row["cu"], "as":row["as"], "cu_method":method, "as_method":method, "status":row["status"]})
    snapshots = output/"metric_snapshots"
    snapshots.mkdir()
    snapshot_count = 0
    def persist_snapshot(snapshot):
        nonlocal snapshot_count
        path = snapshots/(snapshot["id"]+".json")
        if not path.exists():
            write_json(path,snapshot)
            snapshot_count += 1
    with threadpool_limits(limits=1):
        for strategy in protocol["strategies"]:
            progress({"phase":"shadow_replay", "strategy":strategy["id"]})
            clock = time.perf_counter()
            engine = MatureMetrics(forecasts, ledger, attributes, protocol["source_methods"],
                protocol["calibration"]["target_iqr"], protocol["calibration"]["resource_profiles"], RoutingPolicy(**strategy["policy"]))
            router = PolicyRouter(engine,by_mode=strategy["by_mode"],on_snapshot=persist_snapshot)
            samples = []
            for i,event in enumerate(events):
                if i % 100 == 0 and time.perf_counter()-start > max_wall_seconds:
                    pd.DataFrame(records).to_csv(output/"partial_predictions.csv",index=False)
                    with (output/"partial_decisions.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
                        for decision in decisions:
                            stream.write(json.dumps(safe(decision),ensure_ascii=False)+"\n")
                    raise WorkbenchError("回放已到达时间预算，保留已产生的预测和快照", "ROUTING_TIME_BUDGET")
                tick = time.perf_counter()
                chosen = router.decide(event)
                samples.append((time.perf_counter()-tick)*1000)
                records.append({"strategy":strategy["id"],"event_id":event,"decision_at":attributes[event]["decision_at"],
                    "cu":chosen["cu"]["value"],"as":chosen["as"]["value"],
                    "cu_method":chosen["cu"]["selected_method"],"as_method":chosen["as"]["selected_method"],
                    "status":"completed" if all(np.isfinite(chosen[t]["value"]) for t in ("cu","as")) else "failed"})
                decisions.extend({**value,"strategy":strategy["id"]} for value in chosen.values())
                if i % 500 == 0:
                    progress({"phase":"shadow_replay","strategy":strategy["id"],"events_done":i,"total_events":len(events)})
            resources.append({"strategy":strategy["id"],"elapsed_ms":(time.perf_counter()-clock)*1000,
                "routing_p50_ms":float(np.percentile(samples,50)),"routing_p95_ms":float(np.percentile(samples,95)),
                "samples":len(samples),"timing_scope":"routing from already computed forecasts; includes snapshot refresh/persistence",
                "llm_calls":0,"tokens":0,"api_cost_cny":0,"retries":0})
    pd.DataFrame(records).to_csv(output/"predictions.csv",index=False)
    with (output/"decisions.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        for decision in decisions:
            stream.write(json.dumps(safe(decision),ensure_ascii=False,allow_nan=False)+"\n")
    if source_signature(data) != protocol["source"]:
        raise WorkbenchError("回放期间数据来源改变", "SOURCE_CHANGED")
    manifest = {"schema_version":"routing-replay-output.g5a.v1","status":"completed",
        "created_at":utc_now(),"source":protocol["source"],"protocol_sha256":file_hash(output/"protocol.json"),
        "predictions_sha256":file_hash(output/"predictions.csv"),"decisions_sha256":file_hash(output/"decisions.jsonl"),
        "events":len(events),"snapshots":snapshot_count,"resources":resources,
        "elapsed_ms":(time.perf_counter()-start)*1000,"live_model_change":False}
    write_json(output/"replay_manifest.json",manifest)
    progress({"phase":"independent_evaluation"})
    result = evaluate_replay(data,output)
    progress({"phase":"completed","common_events":result["common_events"]})
    return result


def evaluate_replay(data, output):
    output = Path(output)
    protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((output/"replay_manifest.json").read_text(encoding="utf-8"))
    for name,key in (("protocol.json","protocol_sha256"),("predictions.csv","predictions_sha256"),("decisions.jsonl","decisions_sha256")):
        if file_hash(output/name) != manifest[key]:
            raise WorkbenchError("路由回放工件哈希不符", "ROUTING_HASH_MISMATCH")
    ledger = DataService(data).labels
    visible = ledger.latest(protocol["evaluation_as_of"])
    frame = pd.read_csv(output/"predictions.csv")
    strategies = ["fixed_"+m for m in protocol["source_methods"]]+[s["id"] for s in protocol["strategies"]]
    if set(frame.strategy.unique()) != set(strategies):
        raise WorkbenchError("策略结果与预先固定协议不一致", "ROUTING_COVERAGE")
    expected = {(event,strategy) for event in data.fold_for for strategy in strategies}
    if frame.duplicated(["event_id","strategy"]).any() or set(zip(frame.event_id,frame.strategy)) != expected:
        raise WorkbenchError("策略比较缺少完整事件集合", "ROUTING_COVERAGE")
    common = {e for e in data.fold_for if all((e,t) in visible and visible[(e,t)].quality_eligible and visible[(e,t)].value is not None for t in ("cu","as"))}
    coverage = {}
    by_strategy = {}
    for strategy in strategies:
        rows = frame[frame.strategy.eq(strategy)].set_index("event_id")
        good = rows.status.eq("completed") & np.isfinite(rows[["cu","as"]].to_numpy(float)).all(axis=1)
        valid = set(rows.index[good])
        coverage[strategy]={"expected":len(data.fold_for),"valid":len(valid),"failed_or_missing":len(data.fold_for)-len(valid)}
        common &= valid
        by_strategy[strategy]=rows
    events = sorted(common,key=lambda e:(data.row(e).decision_at,e))
    if not events:
        raise WorkbenchError("没有可共同评分的事件", "ROUTING_NO_EVALUATION")
    times = [source_time(data.row(e).decision_at).isoformat() for e in events]
    metrics=[]
    for strategy,rows in by_strategy.items():
        for target,unit in (("cu","g/L"),("as","mg/L")):
            truth=np.array([visible[(e,target)].value for e in events])
            predicted=rows.loc[events,target].to_numpy(float)
            baseline=by_strategy["fixed_Persistence"].loc[events,target].to_numpy(float)
            difference=np.abs(predicted-truth)-np.abs(baseline-truth)
            interval=block_interval(difference,times,RoutingPolicy(block_days=7))
            groups={}
            for mode in sorted({data.mode_cards[e]["mode_code"] for e in events}):
                mask=np.array([data.mode_cards[e]["mode_code"]==mode for e in events])
                groups[mode]=numerical_metrics(truth[mask],predicted[mask])
            folds={}
            for fold in sorted({data.fold_for[e] for e in events}):
                mask=np.array([data.fold_for[e]==fold for e in events])
                folds[fold]=numerical_metrics(truth[mask],predicted[mask])
            metrics.append({"strategy":strategy,"target":target,"unit":unit,**numerical_metrics(truth,predicted),
                "paired_mae_difference":float(difference.mean()),"paired_week_ci95":interval,"mode_metrics":groups,
                "fold_metrics":folds,"negative_predictions":int((predicted<0).sum())})
    decisions=[json.loads(line) for line in (output/"decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    routing={}
    for strategy in [s["id"] for s in protocol["strategies"]]:
        rows=[d for d in decisions if d["strategy"]==strategy]
        routing[strategy]={"reasons":dict(Counter(d["reason"] for d in rows)),
            "selected_models":{t:dict(Counter(d["selected_method"] for d in rows if d["target"]==t)) for t in ("cu","as")},
            "switches":sum(d["reason"]=="SHADOW_SWITCH" for d in rows)}
    result=safe({"schema_version":"routing-evaluation.g5a.v1","status":"completed","created_at":utc_now(),
        "evaluation_mode":"historical_replay","common_events":len(events),"common_event_hash":digest(events),
        "coverage":coverage,"metrics":metrics,"routing":routing,"resources":manifest["resources"],
        "live_model_change":False,"optimization_proxy_approval":False,"external_2026_read":False,
        "protocol_sha256":manifest["protocol_sha256"],"replay_manifest_sha256":file_hash(output/"replay_manifest.json"),
        "evaluator_sha256":file_hash(Path(__file__))})
    if source_signature(data) != protocol["source"]:
        raise WorkbenchError("独立评价的数据版本改变", "SOURCE_CHANGED")
    write_json(output/"evaluation.json",result)
    report="# G5a 成熟标签与影子选模回放\n\n"
    report+=f"共同评价 {len(events)} 个开发期 OOF 事件。参数在回放前固定；Cu、As 分别选模，线上默认模型未改变。\n\n"
    report+="| 策略 | 目标 | MAE | RMSE | 相对 Persistence 的 MAE 差 |\n|---|---|---:|---:|---:|\n"
    for row in metrics:
        report+=f"| {row['strategy']} | {row['target']} ({row['unit']}) | {row['mae']:.6g} | {row['rmse']:.6g} | {row['paired_mae_difference']:.6g} |\n"
    report+="\n最终评价统一使用七天时间块重采样；路由的一天与七天块方案作为预先声明的敏感性对照。时间块独立性是区间计算的假设。\n\n"
    report+="[完整指标与选择理由](evaluation.json) · [预先固定协议](protocol.json) · [预测账本](predictions.csv) · [逐事件决策](decisions.jsonl)\n\n"
    report+="本次是历史策略开发回放。嵌套 OOF 集成、发布/回退与优化器组合仍属于后续 G5 工作。\n"
    (output/"report.md").write_bytes(report.encode("utf-8"))
    return result
