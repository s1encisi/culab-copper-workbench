"""Prepare isolated local document dependencies and pinned public embedding weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["onnxruntime==1.30.0", "tokenizers==0.23.2", "pypdf==6.18.1",
            "python-docx==1.2.0", "lxml==6.1.3", "flatbuffers==25.12.19", "protobuf==6.33.5"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dependencies", action="store_true")
    args = parser.parse_args()
    if not args.skip_dependencies:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
                        "--target", str(ROOT / "runs/dependencies/knowledge-v1"), *PACKAGES], check=True)
    manifest = json.loads((ROOT / "configs/runtime/knowledge_embedding.json").read_text(encoding="utf-8"))
    directory = ROOT / "runs/dependencies/bge-small-zh-v1.5"
    for filename, entry in manifest["files"].items():
        path = directory / filename
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
            continue
        url = f"https://huggingface.co/{manifest['onnx_repository']}/resolve/{manifest['revision']}/{filename}"
        with urlopen(url, timeout=120) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != entry["size"]:
            raise RuntimeError("Public model download does not match the pinned manifest")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(raw)
        temporary.replace(path)
    print(json.dumps({"status": "ready", "model": manifest["model_id"], "revision": manifest["revision"]}))


if __name__ == "__main__":
    main()
