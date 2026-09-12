"""Dedicated BART fitting process; inputs and posterior artifacts stay local."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from copper_mvp.bart_runtime import prepare_bart_runtime
RUNTIME = prepare_bart_runtime()

if __name__ == "__main__":
    import joblib
    import numpy as np
    from copper_mvp.bart_sampling import fit_bart_posterior
    from copper_mvp.bart_registry import bart_source_hashes
    from copper_mvp.common import PROJECT_ROOT, file_hash, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((PROJECT_ROOT/"runs").resolve()):
        parser.error("BART artifacts must remain under project runs/")
    request = json.loads((directory/"request.json").read_text(encoding="utf-8"))
    assert file_hash(directory/"input.npz") == request["input_sha256"]
    assert bart_source_hashes() == request["code_hashes"], "BART source changed before fitting"
    data = np.load(directory/"input.npz", allow_pickle=False)
    try:
        result = fit_bart_posterior(data["X"], data["y"], request["settings"],
                                    request["seed"], directory, RUNTIME)
        assert bart_source_hashes() == request["code_hashes"], "BART source changed during fitting"
        result["code_hashes"] = request["code_hashes"]
        joblib.dump(result, directory/"posterior.joblib", compress=3)
        write_json(directory/"state.json", {"status": "completed",
            "posterior_sha256": file_hash(directory/"posterior.joblib"),
            "diagnostics_passed": result["diagnostics_passed"],
            "elapsed_seconds": result["elapsed_seconds"]})
    except Exception as exc:
        write_json(directory/"state.json", {"status": "failed", "type": type(exc).__name__,
                                           "message": str(exc)[:400]})
        raise
