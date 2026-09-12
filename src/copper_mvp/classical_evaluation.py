"""Independent checks of uncalibrated marginal uncertainty on the common OOF cohort."""
from __future__ import annotations

import numpy as np
import pandas as pd
from copper_mvp.common import WorkbenchError


def uncertainty_metrics(path,labels,common_events):
    frame=pd.read_csv(path)
    if frame.duplicated(["event_id","method_id","target"]).any():
        raise WorkbenchError("概率预测身份重复","UNCERTAINTY_COVERAGE")
    results=[]
    common=set(common_events)
    for (method,target),part in frame.groupby(["method_id","target"]):
        part=part[part.event_id.isin(common)].set_index("event_id")
        if set(part.index)!=common:
            raise WorkbenchError("概率评价未覆盖共同事件集合","UNCERTAINTY_COVERAGE")
        part=part.loc[common_events]
        actual=np.array([labels[(e,target)].value for e in common_events])
        point=part.prediction.to_numpy(float)
        result={"method_id":method,"target":target,"n":len(part),"calibrated":False,"joint_region":False}
        if part.kind.eq("marginal_standard_deviation").all():
            std=part["std"].to_numpy(float)
            if not np.isfinite(std).all() or (std<=0).any():
                raise WorkbenchError("概率标准差必须为有限正数","UNCERTAINTY_VALUES")
            lower=point-1.6448536269514722*std;upper=point+1.6448536269514722*std
            result.update(kind="Gaussian_model_assumption",nominal_coverage=.9,
                empirical_coverage=float(((actual>=lower)&(actual<=upper)).mean()),
                mean_width=float((upper-lower).mean()),
                gaussian_nll=float((.5*np.log(2*np.pi)+np.log(std)+.5*((actual-point)/std)**2).mean()),
                negative_lower_bounds=int((lower<0).sum()))
        elif part.kind.eq("raw_marginal_quantiles").all():
            values=part[["q10","q50","q90"]].to_numpy(float)
            if not np.isfinite(values).all():raise WorkbenchError("分位数输出含非法值","UNCERTAINTY_VALUES")
            valid=(values[:,0]<=values[:,1])&(values[:,1]<=values[:,2])
            residual=actual[:,None]-values
            levels=np.array([.1,.5,.9])
            pinball=np.maximum(levels*residual,(levels-1)*residual)
            covered=valid&(actual>=values[:,0])&(actual<=values[:,2])
            result.update(kind="raw_quantiles",levels=levels.tolist(),nominal_coverage=.8,
                crossing_events=int((~valid).sum()),valid_interval_fraction=float(valid.mean()),
                empirical_coverage_all_events=float(covered.mean()),pinball_mean=pinball.mean(axis=0).tolist(),
                mean_valid_width=float((values[valid,2]-values[valid,0]).mean()) if valid.any() else None,
                interval_scoring_note="Crossed intervals are counted as uncovered; raw quantiles are not silently reordered.")
        else:
            raise WorkbenchError("概率输出类型不一致","UNCERTAINTY_VALUES")
        results.append(result)
    return results
