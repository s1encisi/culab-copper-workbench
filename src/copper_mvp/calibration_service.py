"""Local calibration jobs and artifact-backed historical interval replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import joblib
import numpy as np

from copper_mvp.access import PROJECT
from copper_mvp.model_comparisons import observed_job_state
from copper_mvp.calibration import SplitC90
from copper_mvp.calibration_evaluation import evaluate_calibration,read_calibration_training
from copper_mvp.calibration_training import train_calibration_study
from copper_mvp.common import WorkbenchError,digest,file_hash,utc_now,write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.model_registry import method_spec
from copper_mvp.model_training import source_signature


def load_calibration_model(path,method_id):
    from copper_mvp.model_registry import SPECIALIZED_METHODS,STATISTICAL_METHODS,TABULAR_METHODS,TEMPORAL_METHODS
    if method_id in SPECIALIZED_METHODS:
        from copper_mvp.specialized_models import specialized_dependencies
        specialized_dependencies()
    elif method_id in STATISTICAL_METHODS:
        from copper_mvp.statistical_runtime import statistical_dependencies
        statistical_dependencies()
    elif method_id in TABULAR_METHODS:
        from copper_mvp.tabular_runtime import tabular_runtime
        tabular_runtime()
    elif method_id in TEMPORAL_METHODS:
        from copper_mvp.neural_runtime import load_tensor_runtime
        load_tensor_runtime()
    elif method_id=="BART":
        from copper_mvp.bart_trees import numeric_evaluator
        numeric_evaluator()
    return joblib.load(path)


class CalibrationService:
    def __init__(self,root,data):
        self.root=Path(root)/"calibration_studies";self.root.mkdir(parents=True,exist_ok=True)
        self.data=data;self.lock=threading.RLock()

    @staticmethod
    def authorize(actor,permission="read"):
        actor.require(permission)
        if actor.project_id!=PROJECT:
            raise WorkbenchError("不能访问其他项目的校准研究","FORBIDDEN")

    def directory(self,identifier):
        if len(identifier)!=32 or any(c not in "0123456789abcdef" for c in identifier):
            raise WorkbenchError("校准研究不存在","CALIBRATION_NOT_FOUND")
        return self.root/identifier

    def get(self,actor,identifier):
        self.authorize(actor)
        root=self.directory(identifier)
        if not (root/"state.json").is_file():
            raise WorkbenchError("校准研究不存在","CALIBRATION_NOT_FOUND")
        state=observed_job_state(json.loads((root/"state.json").read_text(encoding="utf-8")))
        if (root/"evaluation.json").is_file():
            state["result"]=json.loads((root/"evaluation.json").read_text(encoding="utf-8"))
        return state

    def list(self,actor):
        self.authorize(actor)
        return [observed_job_state(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.root.glob("*/state.json"),key=lambda p:p.stat().st_mtime,reverse=True)[:100]]

    def submit(self,actor,request,executor):
        self.authorize(actor,"compute")
        identifier=digest({"owner":actor.user_id,"request_key":request.request_key})[:32]
        root=self.directory(identifier)
        fingerprint=digest({"request":request.model_dump(mode="json"),"source":source_signature(self.data)})
        with self.lock:
            if (root/"state.json").exists():
                value=self.get(actor,identifier)
                if value["fingerprint"]!=fingerprint:
                    raise WorkbenchError("请求键已用于不同校准设置","REQUEST_CONFLICT")
                return {**value,"reused":True}
            root.mkdir(parents=True,exist_ok=True)
            write_json(root/"state.json",{"id":identifier,"status":"queued","owner_id":actor.user_id,"project_id":actor.project_id,
                "owner_pid":os.getpid(),"fingerprint":fingerprint,"request":request.model_dump(mode="json"),"created_at":utc_now()})
            executor.submit(self.execute,identifier,request)
        return {**self.get(actor,identifier),"reused":False}

    def execute(self,identifier,request):
        root=self.directory(identifier)
        def update(**values):
            with self.lock:
                state=observed_job_state(json.loads((root/"state.json").read_text(encoding="utf-8")));state.update(values)
                write_json(root/"state.json",state)
        update(status="running",started_at=utc_now())
        try:
            train_calibration_study(self.data,request,root,progress=lambda value:update(progress=value))
            update(progress={"phase":"independent_evaluation"})
            result=evaluate_calibration(self.data,root)
            update(status=result["status"],finished_at=utc_now(),evaluation_sha256=file_hash(root/"evaluation.json"),progress={"phase":"completed"})
        except Exception as error:
            update(status="failed",finished_at=utc_now(),error={"code":getattr(error,"code",type(error).__name__),"message":str(error)[:800]})

    def predict(self,actor,identifier,event_id,method_id,seed=None):
        self.authorize(actor)
        state=self.get(actor,identifier)
        if state["status"]!="completed":
            raise WorkbenchError("校准研究尚未完成","CALIBRATION_STATE")
        root=self.directory(identifier);manifest,protocol=read_calibration_training(root)
        if source_signature(self.data)!=protocol["source"]:
            raise WorkbenchError("校准数据版本已改变","SOURCE_CHANGED")
        if event_id not in self.data.fold_for:
            raise WorkbenchError("该事件没有独立外层回放折","CALIBRATION_EVENT")
        available=protocol["expected_method_seeds"].get(method_id)
        if not available:
            raise WorkbenchError("该研究未包含所选方法","METHOD_NOT_FOUND")
        seed=available[0] if seed is None else seed
        if seed not in available:
            raise WorkbenchError("所选种子不存在","CALIBRATION_SEED")
        fold=self.data.fold_for[event_id]
        artifact=next((item for item in manifest["artifacts"] if item["fold_id"]==fold and item["method_id"]==method_id and item["seed"]==seed),None)
        if not artifact or artifact["status"]!="completed":
            raise WorkbenchError("该方法在此折没有合格校准工件","CALIBRATION_ARTIFACT")
        if source_time(artifact["calibration_fit_cutoff_at"])>source_time(self.data.row(event_id).decision_at):
            raise WorkbenchError("校准器晚于事件决策时间","CALIBRATION_TIME")
        for path_key,hash_key in (("model_path","model_sha256"),("calibrator_path","calibrator_sha256")):
            path=(root/artifact[path_key]).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path)!=artifact[hash_key]:
                raise WorkbenchError("校准工件完整性校验失败","CALIBRATION_HASH")
        if method_spec(method_id,seed)["content_hash"]!=artifact["model_spec_hash"]:
            raise WorkbenchError("模型定义或运行依赖版本已改变","SOURCE_CHANGED")
        model=load_calibration_model(root/artifact["model_path"],method_id)
        calibrator=SplitC90.from_manifest(json.loads((root/artifact["calibrator_path"]).read_text(encoding="utf-8")))
        point=model.predict_context(self.data,[event_id])
        intervals=calibrator.intervals(point)
        return {"study_id":identifier,"event_id":event_id,"method_id":method_id,"seed":seed,"fold_id":fold,
                "point":{"cu":float(point[0,0]),"as":float(point[0,1])},"units":{"cu":"g/L","as":"mg/L"},
                "decision_at":source_time(self.data.row(event_id).decision_at).isoformat(),"computed_at":utc_now(),
                "dataset_version":protocol["source"]["dataset_version"],
                "intervals":{mode:{target:{"lower":float(bounds[0,0,j]),"upper":float(bounds[0,1,j]),
                            "unit":"g/L" if target=="cu" else "mg/L"} for j,target in enumerate(("cu","as"))}
                             for mode,bounds in intervals.items()},
                "calibrator":calibrator.manifest(),"model_sha256":artifact["model_sha256"],
                "calibrator_sha256":artifact["calibrator_sha256"],"base_fit_cutoff_at":artifact["base_fit_cutoff_at"],
                "calibration_fit_cutoff_at":artifact["calibration_fit_cutoff_at"],"evaluation_mode":"historical_oof_replay",
                "automatic_promotion":False,"causal_control":False}
