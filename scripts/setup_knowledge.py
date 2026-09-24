"""Prepare isolated local document dependencies and pinned public retrieval models."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "onnxruntime==1.30.0",
    "tokenizers==0.23.2",
    "pypdf==6.18.1",
    "python-docx==1.2.0",
    "lxml==6.1.3",
    "flatbuffers==25.12.19",
    "protobuf==6.33.5",
    "rapidocr-onnxruntime==1.4.4",
    "pypdfium2==5.13.0",
    "opencv-python-headless==4.10.0.84",
    "pyclipper==1.4.0",
    "shapely==2.1.2",
]


def fetch(manifest, directory):
    repository = manifest.get("onnx_repository", manifest["model_id"])
    for filename, entry in manifest["files"].items():
        path = directory / filename
        if not path.resolve().is_relative_to(directory.resolve()):
            raise RuntimeError("Model manifest contains an invalid local path")
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError("Existing model does not match the pinned manifest")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".download")
        url = f"https://huggingface.co/{repository}/resolve/{manifest['revision']}/{filename}"
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == entry["size"]:
                break
            request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urlopen(request, timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    with partial.open("ab" if append else "wb") as stream:
                        while data := response.read(1024 * 1024):
                            stream.write(data)
                            stream.flush()
                if partial.stat().st_size == entry["size"]:
                    break
            except (OSError, http.client.IncompleteRead):
                if attempt == 2:
                    raise RuntimeError("Model download interrupted; resumable partial retained") from None
        if not partial.exists() or partial.stat().st_size != entry["size"]:
            raise RuntimeError("Model download incomplete; resumable partial retained")
        if hashlib.sha256(partial.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("Downloaded model does not match the pinned hash")
        partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dependencies", action="store_true")
    args = parser.parse_args()
    if not args.skip_dependencies:
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-compile",
                "--timeout",
                "90",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(ROOT / "runs/dependencies/knowledge-v1"),
                *PACKAGES,
            ],
            check=True,
        )
    for config, folder in [
        ("knowledge_embedding.json", "bge-small-zh-v1.5"),
        ("knowledge_reranker.json", "mmarco-minilm-reranker"),
    ]:
        manifest = json.loads((ROOT / "configs/runtime" / config).read_text(encoding="utf-8"))
        fetch(manifest, ROOT / "runs/dependencies" / folder)
        print(json.dumps({"status": "ready", "model": manifest["model_id"], "revision": manifest["revision"]}))


if __name__ == "__main__":
    main()
