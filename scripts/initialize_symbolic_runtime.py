"""Resolve and record the local Julia environment without sending research data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from copper_mvp.common import file_hash, write_json
from copper_mvp.symbolic_runtime import configure_symbolic_runtime

if __name__ == "__main__":
    runtime = configure_symbolic_runtime(allow_install=True)
    import juliapkg

    juliapkg.require_julia("=1.11.9")
    juliapkg.add("SymbolicRegression", uuid="8254be44-1295-4e6a-a16d-46603ac705cb", version="=1.11.3")
    juliapkg.add("PythonCall", uuid="6099a3de-0909-46bc-b1f4-468b9a2dfc0d", version="=0.9.26")
    import pysr
    from juliacall import Main as jl

    julia_version = str(jl.seval("VERSION"))
    symbolic_version = str(jl.seval("pkgversion(SymbolicRegression)"))
    assert julia_version == "1.11.9"
    assert symbolic_version == "1.11.3"
    manifest = runtime["project"] / "Manifest.toml"
    project = runtime["project"] / "Project.toml"
    record = {
        "status": "initialized",
        "pysr": pysr.__version__,
        "julia": julia_version,
        "symbolic_regression": symbolic_version,
        "manifest_sha256": file_hash(manifest),
        "project_sha256": file_hash(project),
        "research_data_sent": False,
    }
    write_json(runtime["project"] / "runtime_lock.json", record)
    print(json.dumps(record), flush=True)
