"""Separate scikit-learn/TabPFN process environment with local-only weights."""
from pathlib import Path
from importlib import metadata
import os,sys
from copper_mvp.common import PROJECT_ROOT,WorkbenchError

PACKAGES={"tabpfn":"2.1.3","scikit-learn":"1.6.1","einops":"0.8.1","huggingface-hub":"0.28.1"}


def prepare_tabpfn_runtime():
    directory=Path(os.environ.get("COPPER_TABPFN_RUNTIME_DIR",PROJECT_ROOT/"runs/dependencies/tabpfn-2.1.3"))
    tensor=Path(os.environ.get("COPPER_TENSOR_RUNTIME_DIR",directory.parent/"torch-botorch-2.8.0"))
    for path in (tensor,directory):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    for name,expected in PACKAGES.items():
        if metadata.version(name)!=expected:
            raise WorkbenchError("TabPFN 隔离依赖版本不一致: "+name,"TABPFN_DEPENDENCY")
    os.environ.update({"HF_HUB_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1",
                      "DO_NOT_TRACK":"1","TABPFN_DISABLE_TELEMETRY":"1"})
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
    return {name:metadata.version(name) for name in (*PACKAGES,"torch","numpy","scipy")}


def disable_worker_network():
    import socket
    def denied(*args,**kwargs):
        raise RuntimeError("TabPFN inference workers do not permit network connections")
    socket.create_connection=denied
    socket.socket.connect=denied
    socket.socket.connect_ex=denied
