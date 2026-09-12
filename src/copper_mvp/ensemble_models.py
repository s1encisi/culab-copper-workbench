"""Small, auditable ensemble estimators over the existing physical-unit models."""
from __future__ import annotations

from itertools import combinations
import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from copper_mvp.common import WorkbenchError
from copper_mvp.model_adapters import validate_X

BASE_METHODS = ("Persistence", "DeltaRidge", "DeltaHGB")


class EnsembleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = "ensemble.nested.g5b.v1"
    inner_boundaries: tuple[float,...] = (.4,.6,.8,1.)
    base_subsets: tuple[tuple[int,...],...] = ((0,2),(0,1,2))
    ridge_alphas: tuple[float,...] = (.1,1.,10.)
    convex_penalties: tuple[float,...] = (0.,.01,.1,1.)
    dynamic_windows: tuple[int,...] = (60,180)
    dynamic_max_days: int = 90
    dynamic_minimum: int = 30
    mode_shrinkage: int = 30
    block_days: tuple[int,...] = (1,7)
    bag_members: tuple[int,...] = (3,5)
    selection_tolerance: float = .01


def training_scale(y):
    values=np.asarray(y,float)
    iqr=np.percentile(values,75,axis=0)-np.percentile(values,25,axis=0)
    std=np.std(values,axis=0)
    scale=np.where(iqr>0,iqr,np.where(std>0,std,1.))
    method=np.where(iqr>0,"IQR",np.where(std>0,"STD_FOR_ZERO_IQR","UNIT_FOR_CONSTANT_TARGET"))
    return scale,method.tolist()


def simplex_weights(predictions,actual,scale,penalty=0.,prior=None):
    """Solve the 2/3-member convex quadratic by enumerating active simplex faces."""
    P=np.asarray(predictions,float);y=np.asarray(actual,float)
    if P.ndim!=2 or P.shape[1] not in (2,3) or len(P)!=len(y) or not len(y):
        raise WorkbenchError("融合权重需要两或三个基模型的共同样本","ENSEMBLE_WEIGHT_SHAPE")
    if not np.isfinite(P).all() or not np.isfinite(y).all() or not np.isfinite(scale) or scale<=0 or penalty<0:
        raise WorkbenchError("融合输入或训练尺度无效","ENSEMBLE_WEIGHT_VALUES")
    m=P.shape[1]
    prior=np.asarray(prior if prior is not None else [1.,*([0.]*(m-1))])
    A=(P-P[:,[0]])/scale
    b=(y-P[:,0])/scale
    Q=A.T@A/len(A)+penalty*np.eye(m)
    q=A.T@b/len(A)+penalty*prior
    best=None
    for size in range(1,m+1):
        for subset in combinations(range(m),size):
            ids=np.array(subset)
            K=np.block([[Q[np.ix_(ids,ids)],np.ones((size,1))],
                        [np.ones((1,size)),np.zeros((1,1))]])
            solution=np.linalg.lstsq(K,np.r_[q[ids],1.],rcond=None)[0][:size]
            if np.min(solution)<-1e-8 or abs(solution.sum()-1)>1e-7:
                continue
            w=np.zeros(m);w[ids]=np.maximum(0.,solution);w/=w.sum()
            loss=float(np.mean((A@w-b)**2)+penalty*np.sum((w-prior)**2))
            if best is None or loss<best[0]-1e-12:
                best=(loss,w)
    if best is None:
        raise WorkbenchError("非负融合权重求解失败","ENSEMBLE_WEIGHT_FIT")
    return best[1]


def expand_weights(weights,subset):
    result=np.zeros(3)
    result[list(subset)]=weights
    return result


def make_meta(alpha):
    return TransformedTargetRegressor(regressor=Pipeline([
        ("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))]),transformer=StandardScaler())


def checked_output(values,n):
    values=np.asarray(values,float)
    if values.shape!=(n,2) or not np.isfinite(values).all():
        raise WorkbenchError("集成预测不是有限的双目标结果","ENSEMBLE_OUTPUT")
    return values


class StackedPredictor:
    def __init__(self,base_models,meta_models,subsets):
        self.base_models=base_models
        self.meta_models=meta_models
        self.subsets=subsets

    def predict(self,X):
        X=validate_X(X)
        used=set().union(*map(set,self.subsets))
        P={j:self.base_models[j].predict(X) for j in used}
        output=[]
        for target,(meta,subset) in enumerate(zip(self.meta_models,self.subsets)):
            Z=np.column_stack([P[j][:,target]-X[:,target] for j in subset])
            output.append(X[:,target]+meta.predict(Z))
        return checked_output(np.column_stack(output),len(X))


class ConvexPredictor:
    def __init__(self,base_models,weights):
        self.base_models=base_models
        self.weights=np.asarray(weights,float)
        if self.weights.shape!=(2,3) or (self.weights<0).any() or not np.allclose(self.weights.sum(axis=1),1):
            raise WorkbenchError("融合权重必须非负且逐目标和为一","ENSEMBLE_WEIGHT_VALUES")

    def predict(self,X):
        X=validate_X(X)
        output=np.zeros((len(X),2))
        for j,model in enumerate(self.base_models):
            if (self.weights[:,j]!=0).any():
                output+=model.predict(X)*self.weights[:,j]
        return checked_output(output,len(X))


class BaggedPredictor:
    def __init__(self,members):
        self.members=members

    def predict(self,X):
        X=validate_X(X)
        return checked_output(np.mean([m.predict(X) for m in self.members],axis=0),len(X))
