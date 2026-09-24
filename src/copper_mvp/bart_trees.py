"""Portable numeric BART trees; no PyMC objects are required for inference."""

import hashlib
import sys
from functools import lru_cache
from importlib import metadata

import numpy as np
from scipy.special import ndtr

from copper_mvp.common import PROJECT_ROOT, WorkbenchError


def export_posterior(collections):
    unique, lookup, draws = [], {}, []
    for collection in collections:
        identifiers = []
        for tree in collection[0]:
            positions = sorted(tree.tree_structure)
            dense = {old: new for new, old in enumerate(positions)}
            features, values, left, right = [], [], [], []
            for position in positions:
                node = tree.tree_structure[position]
                if node.linear_params is not None:
                    raise ValueError("Only constant-leaf BART trees are supported")
                feature = int(node.idx_split_variable)
                if feature >= 0 and tree.split_rules[feature].__name__ != "ContinuousSplitRule":
                    raise ValueError("BART's projected inputs require continuous splits")
                features.append(feature)
                values.append(float(np.asarray(node.value).reshape(-1)[0]))
                left.append(dense[2 * position + 1] if feature >= 0 else -1)
                right.append(dense[2 * position + 2] if feature >= 0 else -1)
            arrays = (
                np.array(features, dtype=np.int32),
                np.array(values, dtype=np.float64),
                np.array(left, dtype=np.int32),
                np.array(right, dtype=np.int32),
            )
            key = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).digest()
            if key not in lookup:
                lookup[key] = len(unique)
                unique.append(arrays)
            identifiers.append(lookup[key])
        draws.append(identifiers)
    offsets = np.r_[0, np.cumsum([len(a[0]) for a in unique])].astype(np.int64)
    features = np.concatenate([a[0] for a in unique])
    values = np.concatenate([a[1] for a in unique])
    left = np.concatenate([np.where(a[2] >= 0, a[2] + offsets[i], -1) for i, a in enumerate(unique)])
    right = np.concatenate([np.where(a[3] >= 0, a[3] + offsets[i], -1) for i, a in enumerate(unique)])
    indices = np.array(draws, dtype=np.int32)
    return {
        "offsets": offsets,
        "features": features,
        "values": values,
        "left": left,
        "right": right,
        "draw_trees": indices,
    }


def _evaluate_numeric_trees(X, offsets, features, values, left, right):
    result = np.empty((len(offsets) - 1, len(X)))
    for tree in range(len(offsets) - 1):
        for row in range(len(X)):
            node = offsets[tree]
            while features[node] >= 0:
                node = left[node] if X[row, features[node]] <= values[node] else right[node]
            result[tree, row] = values[node]
    return result


@lru_cache(maxsize=1)
def numeric_evaluator():
    directories = [PROJECT_ROOT / "runs/dependencies/numba-0.61.2", PROJECT_ROOT / "runs/dependencies/pymc-bart-0.7.0"]
    for directory in directories:
        if directory.is_dir():
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            break
    for name, expected in {"numba": "0.61.2", "llvmlite": "0.44.0"}.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise WorkbenchError("请安装 BART 可选运行时", "BART_DEPENDENCY") from None
        if actual != expected:
            raise WorkbenchError("BART 数值推理依赖版本不一致", "BART_DEPENDENCY")
    from numba import config, njit

    # Numba caches are independent of Python's -B flag. Keep JIT artifacts out
    # of source directories while honoring an explicitly configured cache.
    if not config.CACHE_DIR:
        config.CACHE_DIR = str(PROJECT_ROOT / "runs/bart_runtime_cache/numba")
    return njit(cache=True)(_evaluate_numeric_trees)


def draw_predictions(posterior, X):
    X = np.ascontiguousarray(X, dtype=np.float64)
    values = numeric_evaluator()(X, *(posterior[k] for k in ("offsets", "features", "values", "left", "right")))
    # Add one tree position at a time, avoiding a draws x trees x rows array.
    indices = posterior["draw_trees"]
    draws = np.zeros((len(indices), len(X)))
    for position in range(indices.shape[1]):
        draws += values[indices[:, position]]
    return draws


def mixture_quantiles(means, sigma, levels=(0.1, 0.5, 0.9)):
    sigma = np.asarray(sigma, float)[:, None]
    if sigma.shape[0] != means.shape[0] or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise WorkbenchError("BART 后验噪声与树样本不一致", "BART_POSTERIOR_VALUES")
    lower, upper = (means - 12 * sigma).min(axis=0), (means + 12 * sigma).max(axis=0)
    output = []
    for level in levels:
        low, high = lower.copy(), upper.copy()
        for _ in range(44):
            middle = (low + high) / 2
            cdf = ndtr((middle[None, :] - means) / sigma).mean(axis=0)
            low = np.where(cdf < level, middle, low)
            high = np.where(cdf < level, high, middle)
        output.append((low + high) / 2)
    return np.stack(output, axis=1)
