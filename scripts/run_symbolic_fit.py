"""Dedicated, offline symbolic fitting process with an immutable Julia environment."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

if __name__ == "__main__":
    import numpy as np
    from copper_mvp.common import PROJECT_ROOT,file_hash,write_json
    from copper_mvp.symbolic_runtime import configure_symbolic_runtime
    from copper_mvp.symbolic_registry import symbolic_source_hashes
    from copper_mvp.symbolic_sampling import fit_symbolic_models
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    args=parser.parse_args()
    directory=args.directory.resolve()
    if not directory.is_relative_to((PROJECT_ROOT/"runs").resolve()):
        parser.error("Private fitting artifacts must remain under local runs/")
    request=json.loads((directory/"request.json").read_text(encoding="utf-8"))
    assert file_hash(directory/"input.npz")==request["input_sha256"]
    assert symbolic_source_hashes()==request["code_hashes"]
    runtime=configure_symbolic_runtime()
    lock_path=runtime["project"]/"runtime_lock.json"
    if not lock_path.is_file():raise RuntimeError("Initialize the locked Julia environment first")
    lock=json.loads(lock_path.read_text(encoding="utf-8"))
    for file,key in (("Project.toml","project_sha256"),("Manifest.toml","manifest_sha256")):
        assert file_hash(runtime["project"]/file)==lock[key]
    data=np.load(directory/"input.npz",allow_pickle=False)
    try:
        result=fit_symbolic_models(data["X"],data["y"],request["settings"],request["seed"],directory,lock)
        assert symbolic_source_hashes()==request["code_hashes"]
        result["code_hashes"]=request["code_hashes"]
        write_json(directory/"result.json",result)
        write_json(directory/"state.json",{"status":"completed","result_sha256":file_hash(directory/"result.json"),
                                          "elapsed_seconds":result["elapsed_seconds"]})
    except Exception as exc:
        write_json(directory/"state.json",{"status":"failed","type":type(exc).__name__,"message":str(exc)[:400]})
        raise
