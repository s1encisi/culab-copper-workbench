"""Proper marginal probability scores used by the specialized model study."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm


def gaussian_scores(actual, mean, std):
    actual, mean, std = np.asarray(actual, float), np.asarray(mean, float), np.asarray(std, float)
    if actual.shape != mean.shape or std.shape != mean.shape or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Gaussian scoring requires matching arrays and positive finite scales")
    error = actual-mean
    z = error/std
    density = np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
    crps = std*(z*(2*ndtr(z)-1)+2*density-1/np.sqrt(np.pi))
    alphas = (0.5, 0.2, 0.1)
    total = 0.5*np.abs(error)
    for alpha in alphas:
        radius = norm.ppf(1-alpha/2)*std
        lower, upper = mean-radius, mean+radius
        interval = upper-lower+2/alpha*np.maximum(lower-actual, 0)+2/alpha*np.maximum(actual-upper, 0)
        total += alpha/2*interval
    return {"gaussian_crps": float(crps.mean()), "weighted_interval_score": float((total/(len(alphas)+0.5)).mean()),
            "wis_nominal_coverages": [1-a for a in alphas]}
