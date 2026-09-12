"""Verified source adapters for existing G2a/G5b artifacts and OOF predictions."""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from copper_mvp.common import WorkbenchError,digest,file_hash,safe
from copper_mvp.data_contracts import source_time
from copper_mvp.data import finite
from copper_mvp.model_comparisons import ComparisonService
from copper_mvp.model_evaluation import read_training
from copper_mvp.model_training import load_registered_model,source_signature,safe_artifact_path
from copper_mvp.ensemble_studies import EnsembleStudies
from copper_mvp.ensemble_time import EnsembleData
from copper_mvp.ensemble_models import EnsembleSettings
from copper_mvp.ensemble_fusion import AdaptiveTarget
from copper_mvp.ensemble_evaluation import verify_fold
from copper_mvp.data_service import DataService
from copper_mvp.release_contracts import TARGET_UNITS,TASK_ID


class ArtifactSources:
    def __init__(self,root,data):
        self.root,self.data=Path(root),data
        self.models=ComparisonService(self.root/"model_comparisons",data)
        self.ensembles=EnsembleStudies(self.root,data)
        self.cache={}
        self.dynamic_cache={}

    def source_root(self,kind,identifier):
        service=self.models if kind=="model_comparison" else self.ensembles
        return service.directory(identifier)

    def load(self,actor,request):
        method,target=request.method_id,request.target
        root=self.source_root(request.source_kind,request.source_id)
        if request.source_kind=="model_comparison":
            state=self.models.get(request.source_id)
            if state["status"]!="completed":
                raise WorkbenchError("模型比较尚未完成","ARTIFACT_SOURCE_NOT_READY")
            training,protocol=read_training(root)
            result=state["result"]
            if method not in protocol["request"]["methods"] or request.seed not in (None,protocol["request"]["seed"]):
                raise WorkbenchError("方法或种子不属于该比较","ARTIFACT_SOURCE")
            seed=protocol["request"]["seed"]
            entries=[a for a in training["artifacts"] if a["method_id"]==method and a["status"]=="completed" and a["fold_id"]!="DEVELOPMENT"]
            artifacts=[{"fold_id":a["fold_id"],"path":a["path"],"sha256":a["sha256"],"fit_cutoff_at":a["fit_cutoff_at"],
                        "kind":"registered_model","train_event_hash":a["train_event_hash"]} for a in entries]
            metrics=next(v for v in result["metrics"] if v["method_id"]==method and v["target"]==target)
            baseline=next(v for v in result["metrics"] if v["method_id"]=="Persistence" and v["target"]==target)
            resource=next(v for v in result["resources"] if v["method_id"]==method)
            quality={"mae":metrics["mae"],"baseline_mae":baseline["mae"],"n":metrics["n"],
                "paired_ci95":metrics["paired_mae_ci95"],"negative_predictions":metrics["negative_predictions"],
                "coverage":result["coverage"][method]["coverage"],"p95_ms":resource["single_predict_p95_ms"],
                "mode_metrics":{},"baseline_modes":{}}
            if method == "BART":
                quality["sampling_diagnostics_passed"] = all(a.get("fit_metadata", {}).get("diagnostics_passed", False) for a in entries)
            forecast_table=pd.read_csv(root/"oof_predictions.csv")
            candidate_rows=forecast_table[forecast_table.method_id.eq(method)].set_index("event_id")
            baseline_rows=forecast_table[forecast_table.method_id.eq("Persistence")].set_index("event_id")
            labels=DataService(self.data).labels.latest(protocol["evaluation_as_of"])
            for mode in sorted({v["mode_code"] for v in self.data.mode_cards.values()}):
                events=[e for e in candidate_rows.index if self.data.mode_cards[e]["mode_code"]==mode and (e,target) in labels
                        and labels[(e,target)].quality_eligible and labels[(e,target)].value is not None
                        and np.isfinite(candidate_rows.loc[e,["cu","as"]].to_numpy(float)).all()
                        and np.isfinite(baseline_rows.loc[e,["cu","as"]].to_numpy(float)).all()]
                if events:
                    truth=np.array([labels[(e,target)].value for e in events])
                    quality["mode_metrics"][mode]={"n":len(events),"mae_mean_over_seeds":float(np.abs(candidate_rows.loc[events,target].to_numpy()-truth).mean())}
                    quality["baseline_modes"][mode]={"n":len(events),"mae_mean_over_seeds":float(np.abs(baseline_rows.loc[events,target].to_numpy()-truth).mean())}
            bindings={"protocol.json":file_hash(root/"protocol.json"),"training_manifest.json":file_hash(root/"training_manifest.json"),
                "oof_predictions.csv":training["oof_sha256"],"evaluation.json":state["evaluation_sha256"]}
            family=next(v for v in protocol["methods"] if v["method_id"]==method)
            source=training["source"]
        else:
            state=self.ensembles.get(actor,request.source_id)
            if state["status"]!="completed":
                raise WorkbenchError("集成研究尚未完成","ARTIFACT_SOURCE_NOT_READY")
            protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
            manifest=json.loads((root/"replay_manifest.json").read_text(encoding="utf-8"))
            result=state["result"]
            available=protocol["expected_methods"].get(method)
            if not available or method in protocol["base_methods"] or method in ("ElasticNet","Huber","PLS"):
                raise WorkbenchError("请选择该研究实际生成的集成方法","ARTIFACT_SOURCE")
            seed=request.seed if request.seed is not None else (available[0] if available!=[-1] else None)
            if (seed if seed is not None else -1) not in available:
                raise WorkbenchError("种子不属于该研究","ARTIFACT_SOURCE")
            artifacts=[]
            book=EnsembleData(self.data,DataService(self.data).labels)
            for fold in manifest["folds"]:
                folder=root/fold["directory"]
                expected=next(v for v in protocol["folds"] if v["fold_id"]==fold["fold_id"])
                verify_fold(folder,fold,book,expected)
                entry=next((v for v in fold["artifacts"] if v.get("method",v["name"])==method and v.get("seed")==seed),None)
                if entry is None:raise WorkbenchError("集成模型工件不完整","ARTIFACT_SOURCE")
                artifacts.append({"fold_id":fold["fold_id"],"path":(Path(fold["directory"])/entry["path"]).as_posix(),
                    "sha256":entry["sha256"],"fit_cutoff_at":entry["fit_cutoff_at"],"kind":entry["artifact_type"],
                    "train_event_hash":entry["train_event_hash"]})
            metric=next(v for v in result["metrics"] if v["method_id"]==method and v["target"]==target)
            baseline=next(v for v in result["metrics"] if v["method_id"]=="Persistence" and v["target"]==target)
            coverage=[v["valid_predictions"]/v["expected_events"] for k,v in result["coverage"].items() if k.startswith(method+":")]
            quality={"mae":metric["mae"],"baseline_mae":baseline["mae"],"n":metric["n"],
                "paired_ci95":metric["paired_ci95"],"negative_predictions":metric["negative_predictions"],
                "coverage":min(coverage),"p95_ms":result["resources"][method]["p95_ms"],
                "mode_metrics":metric["mode_metrics"],"baseline_modes":baseline["mode_metrics"],
                "seed_scope":"All predeclared seeds; selecting an artifact seed does not discard adverse seeds."}
            bindings={"protocol.json":file_hash(root/"protocol.json"),"replay_manifest.json":file_hash(root/"replay_manifest.json"),
                "predictions.csv":manifest["predictions_sha256"],"decisions.jsonl":manifest["decisions_sha256"],
                "evaluation.json":state["evaluation_sha256"]}
            family={"method":method,"settings":protocol["settings"],"base_methods":protocol["base_methods"]}
            source=protocol["source"]
        if source_signature(self.data)!=source:
            raise WorkbenchError("当前数据与工件训练来源不同","SOURCE_CHANGED")
        expected_folds=set(self.data.fold_for.values())
        if {a["fold_id"] for a in artifacts}!=expected_folds:
            raise WorkbenchError("缺少原时间折的模型工件","ARTIFACT_SOURCE")
        dependencies={name:metadata.version(name) for name in ("numpy","scikit-learn","joblib")}
        if "statsmodels" in family.get("dependencies",{}):
            from copper_mvp.statistical_runtime import statistical_dependencies
            dependencies.update(statistical_dependencies())
        if method in ("CatBoost", "NGBoost", "EBM", "Cubist"):
            from copper_mvp.specialized_models import specialized_dependencies
            specialized_dependencies()
            dependencies.update({name: metadata.version(name) for name in family["dependencies"]})
        if method == "BART":
            from copper_mvp.bart_trees import numeric_evaluator
            numeric_evaluator()
            dependencies.update({name: metadata.version(name) for name in family["dependencies"]})
        if method in ("TabNet", "FTTransformer", "NODE"):
            from copper_mvp.tabular_runtime import tabular_runtime
            tabular_runtime()
            dependencies.update({name: metadata.version(name) for name in family["dependencies"]})
        descriptor=safe({"schema_version":"model-artifact-set.g5c.v1","source_kind":request.source_kind,"source_id":request.source_id,
            "method_id":method,"target":target,"unit":TARGET_UNITS[target],"seed":seed,"task_id":TASK_ID,
            "feature_columns":self.data.feature_columns,"feature_spec_hash":digest(self.data.feature_columns),
            "source":source,"bindings":bindings,"artifacts":artifacts,"benchmark":quality,
            "family":family,"dependencies":dependencies,
            "license":{"project":"Not declared in the project metadata; no new license is assigned here",
                "dependencies":{name:(metadata.metadata(name).get("License-Expression") or metadata.metadata(name).get("License") or "See installed distribution metadata")[:160]
                    for name in dependencies}},
            "calibrator":None,"calibrator_missing_reason":"No prediction-interval calibration was fitted for these artifacts",
            "scopes":["historical_oof"],"proxy_approved":False,"causal_control":False,
            "evaluation_mode":"historical_replay","benchmark_event_hash":result["common_event_hash"]})
        descriptor["source_fingerprint"]=digest(descriptor)
        self.verify(descriptor)
        return descriptor

    def verify(self,descriptor):
        root=self.source_root(descriptor["source_kind"],descriptor["source_id"])
        if source_signature(self.data)!=descriptor["source"]:
            raise WorkbenchError("工件的数据来源已变化","SOURCE_CHANGED")
        for name,expected in descriptor["bindings"].items():
            if file_hash(safe_artifact_path(root,name))!=expected:
                raise WorkbenchError("工件来源记录已变化","MODEL_HASH_MISMATCH")
        for artifact in descriptor["artifacts"]:
            if file_hash(safe_artifact_path(root,artifact["path"]))!=artifact["sha256"]:
                raise WorkbenchError("模型文件已变化","MODEL_HASH_MISMATCH")
        if descriptor["method_id"] in ("CatBoost", "NGBoost", "EBM", "Cubist"):
            from copper_mvp.specialized_models import specialized_dependencies
            specialized_dependencies()
        if descriptor["method_id"] == "BART":
            from copper_mvp.bart_trees import numeric_evaluator
            numeric_evaluator()
        if descriptor["method_id"] in ("TabNet", "FTTransformer", "NODE"):
            from copper_mvp.tabular_runtime import tabular_runtime
            tabular_runtime()
        for name,version in descriptor["dependencies"].items():
            if metadata.version(name)!=version:
                raise WorkbenchError("模型依赖版本与登记记录不一致","MODEL_VERSION_MISMATCH")
        return True

    def event(self,event_id):
        row=self.data.row(event_id)
        snapshot=DataService(self.data).snapshot(event_id)
        return {"event_id":event_id,"decision_at":source_time(row.decision_at).isoformat(),"snapshot_id":snapshot["id"],
                "dataset_version":self.data.dataset_version,"current":{"cu":finite(row.origin_cu_g_l),"as":finite(row.origin_as_mg_l)}}

    def forecasts(self,descriptor):
        root=self.source_root(descriptor["source_kind"],descriptor["source_id"])
        file="oof_predictions.csv" if descriptor["source_kind"]=="model_comparison" else "predictions.csv"
        frame=pd.read_csv(root/file)
        rows=frame[frame.method_id.eq(descriptor["method_id"])]
        if "seed" in rows:
            rows=rows[rows.seed.isna()] if descriptor["seed"] is None else rows[rows.seed.eq(descriptor["seed"])]
        return rows.set_index("event_id")

    def predict(self,descriptor,event):
        fold=self.data.fold_for.get(event)
        entry=next((a for a in descriptor["artifacts"] if a["fold_id"]==fold),None)
        if entry is None:
            raise WorkbenchError("该事件没有对应 OOF 工件","NO_OOF_MODEL")
        if source_time(entry["fit_cutoff_at"])>source_time(self.data.row(event).decision_at):
            raise WorkbenchError("模型训练晚于该事件","FUTURE_MODEL")
        root=self.source_root(descriptor["source_kind"],descriptor["source_id"])
        path=safe_artifact_path(root,entry["path"])
        if file_hash(path)!=entry["sha256"]:
            raise WorkbenchError("模型工件哈希不符","MODEL_HASH_MISMATCH")
        if entry["kind"]=="stateful_replay":
            replay=self.dynamic_replay(descriptor,entry)
            target=("cu","as").index(descriptor["target"])
            return replay["values"][event][target],{"artifact_sha256":entry["sha256"],"execution":"recomputed_stateful_replay",
                "model_elapsed_ms":replay["event_ms"][event],"replay_total_ms":replay["total_ms"]}
        key=entry["sha256"]
        if key not in self.cache:
            if descriptor["source_kind"]=="model_comparison":
                manifest,protocol=read_training(root)
                self.cache[key]=load_registered_model(root,manifest,protocol,descriptor["method_id"],fold)[0]
            else:
                self.cache[key]=joblib.load(path)
        values=self.cache[key].predict_context(self.data,[event]) if descriptor["source_kind"]=="model_comparison" else self.cache[key].predict(self.data.X.loc[[event]].to_numpy(float))
        return float(values[0,("cu","as").index(descriptor["target"])]),{"artifact_sha256":key,"execution":"model_inference"}

    def dynamic_replay(self,descriptor,entry):
        root=self.source_root(descriptor["source_kind"],descriptor["source_id"])
        manifest=json.loads((root/"replay_manifest.json").read_text(encoding="utf-8"))
        fold=next(f for f in manifest["folds"] if f["fold_id"]==entry["fold_id"])
        source=next(a for a in fold["artifacts"] if a["name"]=="OOFConvex")
        parent_path=safe_artifact_path(root,(Path(fold["directory"])/source["path"]).as_posix())
        if file_hash(parent_path)!=source["sha256"]:
            raise WorkbenchError("动态融合基模型工件已变化","MODEL_HASH_MISMATCH")
        key=(entry["sha256"],source["sha256"])
        if key in self.dynamic_cache:return self.dynamic_cache[key]
        state=json.loads(safe_artifact_path(root,entry["path"]).read_text(encoding="utf-8"))
        base=joblib.load(parent_path).base_models
        plan=json.loads((root/fold["directory"]/"inner_plan.json").read_text(encoding="utf-8"))
        events=plan["outer_validation_ids"]
        book=EnsembleData(self.data,DataService(self.data).labels)
        initial=np.asarray(state["initial_history_predictions"])
        mixers=[AdaptiveTarget(book,t,state["parameters"][t],state["initial_weights"][t],state["scale"][t],
                    [(e,initial[i,:,t]) for i,e in enumerate(state["initial_history_event_ids"])],EnsembleSettings(**state["settings"])) for t in (0,1)]
        start=time.perf_counter()
        predictions=np.stack([m.predict(self.data.X.loc[events].to_numpy(float)) for m in base],axis=1)
        values={};event_ms={}
        for i,event in enumerate(events):
            tick=time.perf_counter()
            values[event]=[mixers[t].predict(event,predictions[i,:,t],evidence=False)[0] for t in (0,1)]
            event_ms[event]=(time.perf_counter()-tick)*1000
        result={"values":values,"event_ms":event_ms,"total_ms":(time.perf_counter()-start)*1000}
        self.dynamic_cache[key]=result
        return result

    def probe(self,descriptor,samples_per_fold):
        self.verify(descriptor)
        forecasts=self.forecasts(descriptor)
        rows=[];timings=[]
        for fold in sorted(set(self.data.fold_for.values())):
            events=forecasts.loc[forecasts.fold_id.eq(fold)].sort_values("decision_at").index.tolist()
            positions=np.unique(np.linspace(0,len(events)-1,min(samples_per_fold,len(events))).astype(int))
            for position in positions:
                event=events[position]
                start=time.perf_counter();value,evidence=self.predict(descriptor,event)
                elapsed=(time.perf_counter()-start)*1000;timings.append(evidence.get("model_elapsed_ms",elapsed))
                expected=float(forecasts.loc[event,descriptor["target"]])
                valid=np.isfinite(value) and np.isclose(value,expected,rtol=1e-10,atol=1e-9)
                rows.append({"event_id":event,"fold_id":fold,"value":value,"expected":expected,"parity":bool(valid),"elapsed_ms":elapsed,**evidence})
        self.verify(descriptor)
        return {"scope":"historical_runtime_shadow","independent_new_labels":False,"samples":len(rows),
            "parity_passed":all(r["parity"] for r in rows),"p50_ms":float(np.percentile(timings,50)),
            "p95_ms":float(np.percentile(timings,95)),"rows":rows,
            "note":"Runtime verification of frozen OOF artifacts; it does not add independent accuracy evidence."}
