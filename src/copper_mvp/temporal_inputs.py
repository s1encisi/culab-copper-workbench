"""Causal seven-anchor process windows with explicit observation ages and masks."""
import numpy as np
from sklearn.impute import SimpleImputer
from copper_mvp.classical_models import StableNumericScale
from copper_mvp.common import WorkbenchError

OFFSETS=(12,10,8,6,4,2,0)


def raw_anchor_windows(data,event_ids):
    frame=data.frame.loc[event_ids]
    values=[];ages=[]
    for offset in OFFSETS:
        columns=[tag+f"__t_minus_{offset}h" for tag in data.tags]
        values.append(frame[columns].to_numpy(float))
        ages.append(frame[[c+"__age_h" for c in columns]].to_numpy(float))
    values=np.stack(values,axis=1)
    ages=np.stack(ages,axis=1)
    if np.any(np.isfinite(values)&np.isfinite(ages)&(ages<0)):
        raise WorkbenchError("过程窗口包含晚于锚点的观测","FUTURE_SEQUENCE_SOURCE")
    missing=~np.isfinite(values)|~np.isfinite(ages)|(ages>2.)
    values[missing]=np.nan
    observed_gap=np.zeros_like(ages)
    pairs=~missing[:,1:]&~missing[:,:-1]
    gap=2.+ages[:,:-1]-ages[:,1:]
    if np.any(pairs&(gap < -1e-9)):
        raise WorkbenchError("过程观测时间逆序","SEQUENCE_SOURCE_ORDER")
    observed_gap[:,1:]=np.where(pairs,np.maximum(0.,gap),0.)
    static=data.X.loc[event_ids].to_numpy(float)[:,:2]
    categories=data.X.loc[event_ids].to_numpy(float)[:,110:].astype(np.int64)
    return values,missing,ages,observed_gap,static,categories


class TemporalInputs:
    def fit(self,data,event_ids):
        values,missing,ages,gaps,static,categories=raw_anchor_windows(data,event_ids)
        self.signal_tags=tuple(data.tags)
        flat=values.reshape(-1,values.shape[-1])
        self.imputer=SimpleImputer(strategy="median",keep_empty_features=True)
        self.scaler=StableNumericScale().fit(self.imputer.fit_transform(flat))
        self.static_scaler=StableNumericScale().fit(static)
        return self

    def transform(self,data,event_ids):
        if tuple(data.tags)!=self.signal_tags:
            raise WorkbenchError("过程序列信号合同不同","SEQUENCE_FEATURE_SCHEMA")
        values,missing,ages,gaps,static,categories=raw_anchor_windows(data,event_ids)
        shape=values.shape
        scaled=self.scaler.transform(self.imputer.transform(values.reshape(-1,shape[-1]))).reshape(shape)
        age=np.where(missing,2.,ages)/2.
        relative=np.broadcast_to((-np.array(OFFSETS)/12.)[None,:,None],(len(values),7,1))
        sequence=np.concatenate((scaled,missing.astype(float),age,gaps/4.,relative),axis=2)
        context=np.column_stack((self.static_scaler.transform(static),np.eye(3)[categories].reshape(len(values),-1)))
        return np.ascontiguousarray(sequence,dtype=np.float32),np.ascontiguousarray(context,dtype=np.float32)
