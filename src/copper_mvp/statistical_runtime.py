"""Load the hash-pinned optional statistics packages from the local project runtime."""
from importlib import metadata
from pathlib import Path
import sys
from copper_mvp.common import PROJECT_ROOT,WorkbenchError

STATISTICAL_RUNTIME=PROJECT_ROOT/"runs"/"dependencies"/"statsmodels-0.14.6"


def activate_statistical_path():
    if STATISTICAL_RUNTIME.is_dir() and str(STATISTICAL_RUNTIME) not in sys.path:
        sys.path.insert(0,str(STATISTICAL_RUNTIME))


def statistical_dependencies():
    activate_statistical_path()
    expected={"statsmodels":"0.14.6","patsy":"1.0.1"}
    for name,version in expected.items():
        try:actual=metadata.version(name)
        except metadata.PackageNotFoundError:
            raise WorkbenchError("统计依赖未安装，请运行项目统计依赖安装脚本","STATISTICAL_DEPENDENCY") from None
        if actual!=version:
            raise WorkbenchError("统计依赖版本与协议不同: "+name,"STATISTICAL_DEPENDENCY")
    return expected
