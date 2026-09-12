"""Load the pinned tabular libraries on top of the project's tensor runtime."""
from importlib import metadata
from pathlib import Path
import os,sys
from copper_mvp.common import PROJECT_ROOT,WorkbenchError
from copper_mvp.neural_runtime import load_tensor_runtime
from copper_mvp.tabular_registry import TABULAR_PACKAGES


def tabular_runtime():
    torch=load_tensor_runtime()
    directory=Path(os.environ.get("COPPER_TABULAR_RUNTIME_DIR",PROJECT_ROOT/"runs/dependencies/tabular-neural-v1"))
    if directory.is_dir() and str(directory) not in sys.path:sys.path.insert(0,str(directory))
    for name,expected in TABULAR_PACKAGES.items():
        try:actual=metadata.version(name)
        except metadata.PackageNotFoundError:raise WorkbenchError("表格神经网络依赖未安装","TABULAR_DEPENDENCY") from None
        if actual!=expected:raise WorkbenchError("表格神经网络依赖版本不一致: "+name,"TABULAR_DEPENDENCY")
    return torch
