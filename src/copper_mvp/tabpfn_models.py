"""Portable TabPFN context artifacts with parent-process preprocessing."""
import hashlib,uuid
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copper_mvp.classical_models import classical_preprocess
from copper_mvp.common import PROJECT_ROOT,WorkbenchError,digest,file_hash
from copper_mvp.tabpfn_registry import TABPFN_PARAMETERS,WEIGHT_SHA256
from copper_mvp.tabpfn_client import request_worker,worker_transaction


class TabPFNModel:
    def __init__(self,seed):
        self.seed=seed
        self.settings=dict(TABPFN_PARAMETERS)
        self.fit_warnings=[]

    def fit(self,X,y,sample_weight=None,max_wall_seconds=None):
        if sample_weight is not None:
            raise WorkbenchError("TabPFN 首版不开放样本权重","MODEL_SAMPLE_WEIGHT_UNSUPPORTED")
        X,y=np.asarray(X,float),np.asarray(y,float)
        if len(X)>self.settings["maximum_context_rows"]:
            raise WorkbenchError("TabPFN 输入超出已声明的上下文范围","TABPFN_CONTEXT_LIMIT")
        self.preprocessor=Pipeline([("columns",classical_preprocess("TabPFN")),
                    ("reduce",PCA(n_components=self.settings["components"],svd_solver="full"))])
        self.train_X=np.ascontiguousarray(self.preprocessor.fit_transform(X),dtype=np.float64)
        self.target_scaler=StandardScaler().fit(y-X[:,:2])
        self.train_y=np.ascontiguousarray(self.target_scaler.transform(y-X[:,:2]),dtype=np.float64)
        self.context_id=hashlib.sha256(self.train_X.tobytes()+self.train_y.tobytes()+
                                     str(self.seed).encode()+WEIGHT_SHA256.encode()+str(self.settings["n_estimators"]).encode()).hexdigest()
        result=self._ensure_context(timeout=max_wall_seconds or 1800)
        self.models=[{"context_id":self.context_id,"weight_sha256":WEIGHT_SHA256}]
        self.fit_metadata={"training_rows":len(X),"projected_features":self.train_X.shape[1],
            "target_delta_mean":self.target_scaler.mean_.tolist(),"target_delta_scale":self.target_scaler.scale_.tolist(),
            "context_id":self.context_id,"weight_sha256":WEIGHT_SHA256,"runtime":result["runtime"],
            "pretrained_weights_changed":False,"network_connections_disabled":True,
            "cpu_context_limit_explicitly_enabled":len(X)>1000,"fit_seconds":result.get("fit_seconds",0.)}
        return self

    def _directory(self):
        root=PROJECT_ROOT/"runs/mvp/tabpfn_contexts"/self.context_id
        root.mkdir(parents=True,exist_ok=True)
        return root

    def _ensure_context(self,timeout=1800):
        root=self._directory();path=root/"train.npz"
        if not path.exists():np.savez(path,X=self.train_X,y=self.train_y)
        return request_worker({"action":"fit","context_id":self.context_id,
            "input_path":str(path),"input_sha256":file_hash(path),
            "n_estimators":self.settings["n_estimators"],"seed":self.seed},timeout=timeout)

    def _predict(self,X):
        X=np.asarray(X,float)
        inputs=np.ascontiguousarray(self.preprocessor.transform(X),dtype=np.float64)
        root=self._directory();identifier=uuid.uuid4().hex
        query=root/(identifier+"_input.npz");result=root/(identifier+"_output.npz")
        np.savez(query,X=inputs)
        with worker_transaction():
            self._ensure_context()
            response=request_worker({"action":"predict","context_id":self.context_id,
                "input_path":str(query),"input_sha256":file_hash(query),"output_path":str(result)})
        assert file_hash(result)==response["sha256"]
        with np.load(result,allow_pickle=False) as output:
            mean=output["mean"].copy();quantiles=output["quantiles"].copy()
        return X[:,:2]+self.target_scaler.inverse_transform(mean),(
            X[:,:2,None].transpose(0,2,1)+self.target_scaler.mean_[None,None,:]
            +quantiles*self.target_scaler.scale_[None,None,:])

    def predict(self,X):
        return self._predict(X)[0]

    def uncertainty(self,X):
        return {"kind":"raw_marginal_quantiles","levels":[.1,.5,.9],"values":self._predict(X)[1],
                "distribution":"pretrained_TabPFN_conditional","calibrated":False,"joint_region":False}
