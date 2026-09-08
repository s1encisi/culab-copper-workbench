"""Start CuLab on loopback. The diagnostic service reads only its DeepSeek key."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.run_dir:
        os.environ["COPPER_MVP_RUN_DIR"] = str(args.run_dir.resolve())
    import uvicorn
    uvicorn.run("copper_mvp.api:create_app", host="127.0.0.1", port=args.port, factory=True, access_log=False)
