"""Pinned local PySR/Julia runtime used only by symbolic fitting workers."""

import os
import sys
from importlib import metadata
from pathlib import Path

from copper_mvp.common import PROJECT_ROOT, WorkbenchError

PYTHON_PACKAGES = {"pysr": "1.5.9", "juliacall": "0.9.26", "juliapkg": "0.1.17", "sympy": "1.14.0", "mpmath": "1.3.0"}


def symbolic_runtime_directory():
    """Resolve the shared Python runtime path without starting Julia."""
    return Path(os.environ.get("COPPER_SYMBOLIC_RUNTIME_DIR", PROJECT_ROOT / "runs/dependencies/pysr-1.5.9")).resolve()


def configure_symbolic_runtime(allow_install=False):
    directory = symbolic_runtime_directory()
    executable = Path(
        os.environ.get("COPPER_JULIA_EXECUTABLE", directory.parent / "julia-runtime/julia-1.11.9/bin/julia.exe")
    ).resolve()
    if not executable.is_file():
        raise WorkbenchError("Julia 1.11.9 运行时尚未安装", "SYMBOLIC_DEPENDENCY")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    for name, version in PYTHON_PACKAGES.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise WorkbenchError("符号回归 Python 依赖尚未安装", "SYMBOLIC_DEPENDENCY") from None
        if actual != version:
            raise WorkbenchError("符号回归依赖版本不同: " + name, "SYMBOLIC_DEPENDENCY")
    project = directory.parent / "symbolic-julia-environment"
    depot = directory.parent / "symbolic-julia-depot"
    project.mkdir(parents=True, exist_ok=True)
    depot.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "PYTHON_JULIAPKG_EXE": str(executable),
            "PYTHON_JULIAPKG_PROJECT": str(project),
            "PYTHON_JULIAPKG_OFFLINE": "no" if allow_install else "yes",
            "JULIA_DEPOT_PATH": str(depot),
            "JULIA_NUM_THREADS": "1",
            "PYTHON_JULIACALL_THREADS": "1",
            "PYTHON_JULIACALL_HANDLE_SIGNALS": "no",
            "JULIA_PKG_PRECOMPILE_AUTO": "0",
            "JULIA_NUM_PRECOMPILE_TASKS": "1",
        }
    )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    return {
        "python_packages": dict(PYTHON_PACKAGES),
        "julia_executable": executable,
        "project": project,
        "depot": depot,
    }
