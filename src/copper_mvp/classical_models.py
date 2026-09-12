"""Train-only classical model adapters with explicit multi-output and uncertainty contracts."""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge,ARDRegression,RANSACRegressor,TheilSenRegressor,QuantileRegressor,MultiTaskElasticNet,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from copper_mvp.common import WorkbenchError
from copper_mvp.classical_registry import PARAMETERS,REDUCTION,JOINT,WEIGHTED


class StableNumericScale(BaseEstimator,TransformerMixin):
    """Freeze numerical-noise columns using training data, preserving feature positions."""
    def fit(self,X,y=None):
        X=np.asarray(X,float)
        self.n_features_in_=X.shape[1]
        self.mean_=X.mean(axis=0)
        self.var_=X.var(axis=0)
        self.tolerance_=64*np.finfo(float).eps*np.maximum(1.,np.abs(X).max(axis=0))
        self.active_=self.var_>self.tolerance_**2
        self.scale_=np.where(self.active_,np.sqrt(self.var_),1.)
        return self

    def transform(self,X):
        X=np.asarray(X,float)
        result=(X-self.mean_)/self.scale_
        result[:,~self.active_]=0.
        return result

    def get_feature_names_out(self,input_features=None):
        return np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)],dtype=object)


def classical_preprocess(method):
    columns=ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True,add_indicator=True)),
                             ("scale",StableNumericScale())]),list(range(110))),
        ("modes",OneHotEncoder(categories=[[0.,1.,2.]]*4,handle_unknown="ignore",sparse_output=False),list(range(110,114)))])
    if method in REDUCTION:
        return Pipeline([("columns",columns),("reduce",PCA(n_components=REDUCTION[method],svd_solver="full"))])
    return columns


def estimator(method,seed,quantile=None):
    p=dict(PARAMETERS[method])
    if method=="BayesianRidge":return BayesianRidge(**p)
    if method=="ARD":return ARDRegression(**p)
    if method=="RANSAC":
        p.pop("estimator")
        return RANSACRegressor(estimator=Ridge(alpha=1.),random_state=seed,**p)
    if method=="TheilSen":return TheilSenRegressor(random_state=seed,**p)
    if method=="Quantile":
        p.pop("quantiles")
        return QuantileRegressor(quantile=quantile if quantile is not None else .5,**p)
    if method=="MultiTaskElasticNet":return MultiTaskElasticNet(**p)
    if method=="KernelRidge":return KernelRidge(**p)
    if method=="SVR":return SVR(**p)
    if method=="KNN":return KNeighborsRegressor(**p)
    if method=="GaussianProcess":
        p.pop("kernel")
        return GaussianProcessRegressor(kernel=ConstantKernel(1.,"fixed")*RBF(1.,"fixed"),random_state=seed,**p)
    if method=="DecisionTree":return DecisionTreeRegressor(random_state=seed,**p)
    if method=="RandomForest":return RandomForestRegressor(random_state=seed,**p)
    if method=="AdaBoost":
        p.pop("estimator")
        return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3,min_samples_leaf=5,random_state=seed),random_state=seed,**p)
    if method=="MLP":
        p["hidden_layer_sizes"]=tuple(p["hidden_layer_sizes"])
        return MLPRegressor(random_state=seed,**p)
    raise WorkbenchError("没有该经典模型","METHOD_NOT_FOUND")


def wrapped_model(method,seed,quantile=None):
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocess",classical_preprocess(method)),("regressor",estimator(method,seed,quantile))]),
        transformer=StandardScaler())


class ClassicalModel:
    def __init__(self,method,seed):
        self.method_id,self.seed=method,seed
        self.models=[];self.quantile_models={};self.fit_warnings=[]

    def fit(self,X,y,sample_weight=None):
        if len(X)<(16 if self.method_id=="KNN" else 10):
            raise WorkbenchError("该方法的训练事件数不足","MODEL_TRAINING_SUPPORT")
        if sample_weight is not None:
            weight=np.asarray(sample_weight,float)
            if self.method_id not in WEIGHTED:
                raise WorkbenchError("该方法未开放训练样本权重","MODEL_SAMPLE_WEIGHT_UNSUPPORTED")
            if weight.shape!=(len(X),) or not np.isfinite(weight).all() or (weight<0).any() or weight.sum()<=0:
                raise WorkbenchError("样本权重必须有限、非负且总和为正","MODEL_SAMPLE_WEIGHT")
            kwargs={"regressor__sample_weight":weight}
        else:
            kwargs={}
        delta=np.asarray(y)-np.asarray(X)[:,:2]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if self.method_id=="Quantile":
                for q in (.1,.5,.9):
                    models=[wrapped_model(self.method_id,self.seed,q) for _ in range(2)]
                    for target,model in enumerate(models):model.fit(X,delta[:,target],**kwargs)
                    self.quantile_models[q]=models
                self.models=self.quantile_models[.5]
            elif self.method_id in JOINT:
                model=wrapped_model(self.method_id,self.seed)
                model.fit(X,delta,**kwargs)
                self.models=[model]
            else:
                self.models=[wrapped_model(self.method_id,self.seed) for _ in range(2)]
                for target,model in enumerate(self.models):model.fit(X,delta[:,target],**kwargs)
        self.fit_warnings=[{"category":w.category.__name__,"message":str(w.message)[:400]} for w in captured]
        return self

    def predict(self,X):
        if self.method_id in JOINT:
            delta=np.asarray(self.models[0].predict(X)).reshape(len(X),2)
        else:
            delta=np.column_stack([model.predict(X) for model in self.models])
        return np.asarray(X)[:,:2]+delta

    def uncertainty(self,X):
        current=np.asarray(X)[:,:2]
        if self.method_id=="Quantile":
            values=np.stack([current+np.column_stack([m.predict(X) for m in self.quantile_models[q]]) for q in (.1,.5,.9)],axis=1)
            crossing=(values[:,0,:]>values[:,1,:]) | (values[:,1,:]>values[:,2,:])
            return {"kind":"raw_marginal_quantiles","levels":[.1,.5,.9],"values":values,
                    "crossings":int(crossing.sum()),"calibrated":False,"joint_region":False}
        if self.method_id not in ("BayesianRidge","ARD","GaussianProcess"):
            raise WorkbenchError("该方法没有经接口验证的概率输出","MODEL_UNCERTAINTY_UNSUPPORTED")
        if self.method_id=="GaussianProcess":
            model=self.models[0]
            transformed=model.regressor_.named_steps["preprocess"].transform(X)
            mean,std=model.regressor_.named_steps["regressor"].predict(transformed,return_std=True)
            mean=current+model.transformer_.inverse_transform(np.asarray(mean).reshape(len(X),2))
            std=np.sqrt(np.asarray(std).reshape(len(X),2)**2+PARAMETERS["GaussianProcess"]["alpha"])*model.transformer_.scale_
        else:
            means=[];stds=[]
            for target,model in enumerate(self.models):
                transformed=model.regressor_.named_steps["preprocess"].transform(X)
                mean,std=model.regressor_.named_steps["regressor"].predict(transformed,return_std=True)
                means.append(current[:,target]+model.transformer_.inverse_transform(mean[:,None])[:,0])
                stds.append(std*model.transformer_.scale_[0])
            mean=np.column_stack(means);std=np.column_stack(stds)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<0).any():
            raise WorkbenchError("概率输出无效","MODEL_UNCERTAINTY_VALUES")
        return {"kind":"marginal_standard_deviation","mean":mean,"std":std,"calibrated":False,"joint_region":False}
