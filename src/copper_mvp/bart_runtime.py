"""Activate the pinned PyMC runtime in its dedicated fitting process."""

import os
import sys
from importlib import metadata

from copper_mvp.bart_registry import BART_PACKAGES, BART_PARAMETERS
from copper_mvp.common import PROJECT_ROOT, WorkbenchError


def prepare_bart_runtime():
    runtime = PROJECT_ROOT / "runs/dependencies/pymc-bart-0.7.0"
    if runtime.is_dir() and str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    for name, expected in BART_PACKAGES.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise WorkbenchError("请运行 scripts/setup_bart_models.ps1", "BART_DEPENDENCY") from None
        if actual != expected:
            raise WorkbenchError("BART 依赖版本不一致: " + name, "BART_DEPENDENCY")
    cache = PROJECT_ROOT / "runs/bart_runtime_cache"
    os.environ["PYTENSOR_FLAGS"] = (
        "cxx=,blas__ldflags=,mode=" + BART_PARAMETERS["pytensor_mode"] + ",base_compiledir=" + str(cache / "pytensor")
    )
    os.environ["NUMBA_CACHE_DIR"] = str(cache / "numba")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    # ArviZ 0.20 requires this retained compatibility module in Matplotlib 3.11.
    return {name: metadata.version(name) for name in (*BART_PACKAGES, "numpy", "scipy", "pandas", "matplotlib")}
