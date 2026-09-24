"""Pinned optional CPU tensor runtime shared by neural and Bayesian methods."""

import os
import sys
from contextlib import contextmanager
from importlib import metadata
from threading import RLock

from copper_mvp.common import PROJECT_ROOT, WorkbenchError

_TENSOR_LOCK = RLock()


def load_tensor_runtime():
    from pathlib import Path

    directory = Path(
        os.environ.get("COPPER_TENSOR_RUNTIME_DIR", PROJECT_ROOT / "runs" / "dependencies" / "torch-botorch-2.8.0")
    )
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    for name, expected in {
        "torch": "2.8.0",
        "botorch": "0.17.2",
        "gpytorch": "1.15.2",
        "linear-operator": "0.6.1",
    }.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise WorkbenchError("请运行 scripts/setup_neural_bo.ps1 安装可选运行时", "TENSOR_DEPENDENCY") from None
        if actual != expected:
            raise WorkbenchError("张量运行时版本与协议不一致: " + name, "TENSOR_DEPENDENCY")
    import torch

    torch.set_num_threads(1)
    return torch


@contextmanager
def tensor_random_scope(torch, seed):
    # All project tensor training/search should use this scope for reproducibility.
    with _TENSOR_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield
