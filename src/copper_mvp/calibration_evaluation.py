"""Independent scoring and provenance checks for saved calibration predictions."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from copper_mvp.calibration import SplitC90,interval_metrics,training_scale
from copper_mvp.calibration_training import visible_labels
from copper_mvp.common import WorkbenchError,digest,file_hash,write_json,utc_now
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_evaluation import numerical_metrics,paired_week_interval
from copper_mvp.model_training import source_signature

TARGETS=("cu","as")


def read_calibration_training(root):
    root=Path(root)
    manifest=json.loads((root/"training_manifest.json").read_text(encoding="utf-8"))
    for name,key in (("protocol.json","protocol_sha256"),("predictions.csv","predictions_sha256"),
                     ("calibration_predictions.csv","calibration_predictions_sha256")):
        if file_hash(root/name)!=manifest[key]:
            raise WorkbenchError("校准工件哈希不一致: "+name,"CALIBRATION_HASH")
    protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    return manifest,protocol


def evaluate_calibration(data,root):
    root=Path(root)
    if (root/"evaluation.json").exists():
        raise WorkbenchError("该校准研究已完成评价","CALIBRATION_EXISTS")
    manifest,protocol=read_calibration_training(root)
    if source_signature(data)!=protocol["source"]:
        raise WorkbenchError("校准研究的数据来源已变更","SOURCE_CHANGED")
    ledger=DataService(data).labels
    splits={s["fold_id"]:s for s in protocol["splits"]}
    audit=[]
    for split in splits.values():
        fit,cal,validation=map(set,(split["fit_ids"],split["calibration_ids"],split["validation_ids"]))
        if fit|cal|set(split["purged_ids"])!=set(split["outer_ids"]):
            raise WorkbenchError("外层训练事件未完整分配","CALIBRATION_SPLIT")
        for field,hash_field in (("fit_ids","fit_event_hash"),("calibration_ids","calibration_event_hash"),("validation_ids","validation_event_hash"),("outer_ids","outer_event_hash")):
            if digest(split[field])!=split[hash_field]:
                raise WorkbenchError("时间段事件哈希不一致","CALIBRATION_SPLIT")
        if split["fold_id"]!="DEVELOPMENT":
            cv=data.cv[data.cv.fold_id.eq(split["fold_id"])]
            if set(split["outer_ids"])!=set(data.training_ids(split["fold_id"])) or validation!=set(cv.loc[cv.fold_role.eq("VALIDATION"),"origin_event_id"]):
                raise WorkbenchError("校准研究改变了原外层时间折","CALIBRATION_SPLIT")
        ordered=sorted(split["outer_ids"],key=lambda event:(source_time(data.row(event).decision_at),event))
        wanted=max(protocol["request"]["minimum_calibration"],math.ceil(len(ordered)*protocol["request"]["calibration_fraction"]))
        if source_time(data.row(ordered[-wanted]).decision_at)!=source_time(split["base_fit_cutoff_at"]):
            raise WorkbenchError("校准段不是协议规定的后续时间段","CALIBRATION_SPLIT")
        if fit&cal or fit&validation or cal&validation:
            raise WorkbenchError("校准数据段相交","CALIBRATION_SPLIT")
        if any(source_time(data.row(e).decision_at)>=source_time(split["base_fit_cutoff_at"]) for e in fit):
            raise WorkbenchError("基础模型拟合时间越界","CALIBRATION_SPLIT")
        if any(not source_time(split["base_fit_cutoff_at"])<=source_time(data.row(e).decision_at)<source_time(split["calibration_fit_cutoff_at"]) for e in cal):
            raise WorkbenchError("校准事件越界","CALIBRATION_SPLIT")
        if any(source_time(data.row(e).decision_at)<source_time(split["calibration_fit_cutoff_at"]) for e in validation):
            raise WorkbenchError("验证事件早于校准截止","CALIBRATION_SPLIT")
        _,fit_hash=visible_labels(ledger,split["fit_ids"],split["base_fit_cutoff_at"])
        _,cal_hash=visible_labels(ledger,split["calibration_ids"],split["calibration_fit_cutoff_at"])
        if fit_hash!=split["fit_labels_hash"] or cal_hash!=split["calibration_labels_hash"]:
            raise WorkbenchError("校准标签版本不一致","CALIBRATION_LABEL_HASH")
        audit.append({"fold_id":split["fold_id"],"fit_events":len(fit),"calibration_events":len(cal),"validation_events":len(validation),"passed":True})
    frame=pd.read_csv(root/"predictions.csv")
    if frame.duplicated(["event_id","method_id","seed"]).any():
        raise WorkbenchError("校准预测身份重复","CALIBRATION_PREDICTIONS")
    cal_frame=pd.read_csv(root/"calibration_predictions.csv")
    if len(cal_frame) and cal_frame.duplicated(["event_id","fold_id","method_id","seed"]).any():
        raise WorkbenchError("校准段预测身份重复","CALIBRATION_PREDICTIONS")
    expected=set(data.fold_for)
    methods=protocol["expected_method_seeds"]
    by_case={}
    coverage=[]
    common=set(expected)
    for method,seeds in methods.items():
        for seed in seeds:
            part=frame[frame.method_id.eq(method)&frame.seed.eq(seed)].set_index("event_id")
            if set(part.index)!=expected:
                raise WorkbenchError("预测未完整记录应评价事件","CALIBRATION_COVERAGE")
            for event,row in part.iterrows():
                split=splits[row.fold_id]
                if row.fold_id!=data.fold_for[event] or source_time(row.decision_at)!=source_time(data.row(event).decision_at):
                    raise WorkbenchError("校准预测时间或归属不一致","CALIBRATION_PREDICTIONS")
                if row.status=="completed" and source_time(row.calibration_fit_cutoff_at)!=source_time(split["calibration_fit_cutoff_at"]):
                    raise WorkbenchError("预测使用的校准截止不一致","CALIBRATION_PREDICTIONS")
            good=part.status.eq("completed")
            for field in ("cu","as","marginal_c90_cu_lower","marginal_c90_cu_upper","marginal_c90_as_lower","marginal_c90_as_upper"):
                good &= np.isfinite(part[field]) if field in part else False
            valid=set(part.index[good])
            common &= valid
            coverage.append({"method_id":method,"seed":seed,"expected":len(expected),"available":len(valid),
                             "missing_or_failed":len(expected)-len(valid),"fraction":len(valid)/len(expected)})
            by_case[(method,seed)]=part
    visible=ledger.latest(source_time(protocol["evaluation_as_of"]))
    eligible={event for event in common if all((event,t) in visible and visible[(event,t)].quality_eligible and visible[(event,t)].value is not None for t in TARGETS)}
    events=sorted(eligible,key=lambda e:(source_time(data.row(e).decision_at),e))
    # Independently reconstruct every saved calibrator from the calibration ledger.
    artifact_checks=[]
    for artifact in manifest["artifacts"]:
        if artifact["status"]!="completed":
            continue
        split=splits[artifact["fold_id"]]
        for path_key,hash_key in (("model_path","model_sha256"),("calibrator_path","calibrator_sha256")):
            path=(root/artifact[path_key]).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path)!=artifact[hash_key]:
                raise WorkbenchError("校准模型或校准器工件不一致","CALIBRATION_HASH")
        saved=json.loads((root/artifact["calibrator_path"]).read_text(encoding="utf-8"))
        part=cal_frame[cal_frame.fold_id.eq(artifact["fold_id"])&cal_frame.method_id.eq(artifact["method_id"])&cal_frame.seed.eq(artifact["seed"])].set_index("event_id")
        if set(part.index)!=set(split["calibration_ids"]):
            raise WorkbenchError("校准器缺少完整校准段预测","CALIBRATION_COVERAGE")
        actual,_=visible_labels(ledger,split["calibration_ids"],split["calibration_fit_cutoff_at"])
        fit_actual,_=visible_labels(ledger,split["fit_ids"],split["base_fit_cutoff_at"])
        scale=training_scale(data.X.loc[split["fit_ids"]].to_numpy(float),fit_actual)
        np.testing.assert_allclose(scale,saved["training_delta_std"],rtol=1e-12,atol=1e-12)
        fitted=SplitC90.fit(part.loc[split["calibration_ids"],list(TARGETS)].to_numpy(float),actual,
                           scale,protocol["request"]["coverage"],
                           protocol["request"]["minimum_calibration"])
        np.testing.assert_allclose(fitted.marginal_radius,saved["marginal_radius"],rtol=1e-12,atol=1e-10)
        if fitted.joint_radius is None:
            if saved["joint_score_radius"] is not None:raise WorkbenchError("联合区域状态不一致","CALIBRATION_REPLAY")
        else:
            np.testing.assert_allclose(fitted.joint_radius,saved["joint_score_radius"],rtol=1e-12,atol=1e-10)
        if artifact["fold_id"]!="DEVELOPMENT":
            expected_bounds=fitted.intervals(by_case[(artifact["method_id"],artifact["seed"])].loc[split["validation_ids"],list(TARGETS)].to_numpy(float))
            valid=by_case[(artifact["method_id"],artifact["seed"])].loc[split["validation_ids"]]
            for mode,bounds in expected_bounds.items():
                for j,target in enumerate(TARGETS):
                    np.testing.assert_allclose(bounds[:,0,j],valid[mode+"_"+target+"_lower"],rtol=1e-12,atol=1e-9)
                    np.testing.assert_allclose(bounds[:,1,j],valid[mode+"_"+target+"_upper"],rtol=1e-12,atol=1e-9)
        artifact_checks.append({"fold_id":artifact["fold_id"],"method_id":artifact["method_id"],"seed":artifact["seed"],"calibrator_reconstructed":True})
    metrics=[];joint_metrics=[];groups=[];wis_arrays={}
    if events:
        actual=np.array([[visible[(event,t)].value for t in TARGETS] for event in events])
        decisions=[source_time(data.row(event).decision_at).isoformat() for event in events]
        for (method,seed),part in by_case.items():
            part=part.loc[events]
            for mode in ("marginal_c90","joint_c90","native"):
                if mode+"_cu_lower" not in part:
                    continue
                lo=part[[mode+"_cu_lower",mode+"_as_lower"]].to_numpy(float)
                hi=part[[mode+"_cu_upper",mode+"_as_upper"]].to_numpy(float)
                valid=np.isfinite(lo).all(axis=1)&np.isfinite(hi).all(axis=1)&(lo<=hi).all(axis=1)
                level=float(part.native_coverage.dropna().iloc[0]) if mode=="native" and "native_coverage" in part and part.native_coverage.notna().any() else protocol["request"]["coverage"]
                covered=valid&((actual>=lo)&(actual<=hi)).all(axis=1)
                joint_metrics.append({"method_id":method,"seed":seed,"mode":mode,"n":len(events),
                    "valid_regions":int(valid.sum()),"joint_picp_all_events":float(covered.mean()),
                    "joint_nominal_coverage":level if mode=="joint_c90" else None,
                    "note":"Marginal nominal levels do not imply joint coverage."})
                for j,target in enumerate(TARGETS):
                    if not valid.any():
                        continue
                    point=part["native_"+target+"_center"].to_numpy(float) if mode=="native" else part[target].to_numpy(float)
                    values=interval_metrics(actual[valid,j],point[valid],lo[valid,j],hi[valid,j],level)
                    values["normalized_width"]=float(np.mean((hi[valid,j]-lo[valid,j])/part[target+"_scale"].to_numpy(float)[valid])) if (part[target+"_scale"].to_numpy(float)[valid]>0).all() else None
                    row={"method_id":method,"seed":seed,"mode":mode,"target":target,"unit":"g/L" if target=="cu" else "mg/L",
                         "common_events":len(events),"invalid_intervals":int((~valid).sum()),**values}
                    row["picp_valid_intervals"]=row["picp"]
                    row["picp"]=float((valid&(actual[:,j]>=lo[:,j])&(actual[:,j]<=hi[:,j])).mean())
                    row["wis_valid_intervals"]=row["wis"]
                    row["interval_score_valid_intervals"]=row["interval_score"]
                    if not valid.all():
                        row["wis"]=None;row["interval_score"]=None
                    row.update(numerical_metrics(actual[valid,j],point[valid]))
                    if mode=="native" and "native_"+target+"_std" in part and part["native_"+target+"_std"].notna().all():
                        from copper_mvp.specialized_evaluation import gaussian_scores
                        row.update(gaussian_scores(actual[valid,j],point[valid],part["native_"+target+"_std"].to_numpy(float)[valid]))
                    metrics.append(row)
                    alpha=1-level;width=hi[:,j]-lo[:,j]
                    score=width+(2/alpha)*(lo[:,j]-actual[:,j])*(actual[:,j]<lo[:,j])+(2/alpha)*(actual[:,j]-hi[:,j])*(actual[:,j]>hi[:,j])
                    wis=(.5*np.abs(actual[:,j]-point)+(alpha/2)*score)/1.5
                    if valid.all():wis_arrays[(method,seed,mode,target)]=wis
                    for fold in sorted(data.cv.fold_id.unique()):
                        mask=valid&part.fold_id.eq(fold).to_numpy()
                        if mask.any():
                            group=interval_metrics(actual[mask,j],point[mask],lo[mask,j],hi[mask,j],level)
                            groups.append({"method_id":method,"seed":seed,"mode":mode,"target":target,"fold_id":fold,**group})
        for row in metrics:
            key=(row["method_id"],row["seed"],row["mode"],row["target"])
            baseline=("Persistence",methods["Persistence"][0],row["mode"],row["target"])
            if key in wis_arrays and baseline in wis_arrays:
                delta=wis_arrays[key]-wis_arrays[baseline]
                row["paired_wis_difference"]=float(delta.mean())
                row["paired_wis_ci95"]=paired_week_interval(delta,decisions,{"minimum_blocks":20,"replicates":1000,"seed":protocol["request"]["seeds"][0]})
    result={"status":"completed" if events else "insufficient_common_coverage","created_at":utc_now(),
            "common_events":len(events),"common_event_hash":digest(events),"coverage":coverage,
            "metrics":metrics,"joint_coverage":joint_metrics,"by_fold":groups,"split_audit":audit,
            "calibrator_replay":artifact_checks,"failed_artifacts":[v for v in manifest["artifacts"] if v["status"]!="completed"],
            "resources":[{k:v for k,v in a.items() if k in ("fold_id","method_id","seed","status","base_fit_ms","calibration_predict_ms","calibrator_fit_ms","validation_predict_ms","single_p50_ms","single_p95_ms","total_ms")} for a in manifest["artifacts"]],
            "empirical_only":True,"unconditional_coverage_guarantee":False,"automatic_promotion":False,
            "protocol_sha256":file_hash(root/"protocol.json"),"predictions_sha256":file_hash(root/"predictions.csv")}
    write_json(root/"evaluation.json",result)
    return result
