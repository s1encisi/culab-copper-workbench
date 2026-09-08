"""为正式工程生成逐文件 SHA-256 清单。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_MANIFEST = PROJECT_ROOT / "provenance" / "file_manifest_v1.json"
SHA_MANIFEST = PROJECT_ROOT / "MANIFEST.sha256"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".pytest_tmp",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "runs",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_RELATIVE_FILES = {
    "MANIFEST.sha256",
    "provenance/file_manifest_v1.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_static_project_file(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    relative_text = relative.as_posix()
    if relative_text in EXCLUDED_RELATIVE_FILES:
        return False
    if path.name.endswith(".tsbuildinfo"):
        return False
    if any(
        part in EXCLUDED_DIR_NAMES
        or part.startswith(".venv")
        or part.endswith(".egg-info")
        for part in relative.parts
    ):
        return False
    if path.name != ".env.example" and (
        path.name == ".env" or path.name.startswith(".env.")
    ):
        return False
    return path.is_file()


def collect_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    files = []
    for directory, subdirs, filenames in os.walk(PROJECT_ROOT):
        subdirs[:] = [name for name in subdirs if name not in EXCLUDED_DIR_NAMES and not name.startswith((".venv", ".pytest_tmp")) and not name.endswith(".egg-info")]
        files.extend(Path(directory) / name for name in filenames)
    for path in sorted(files, key=lambda item: item.as_posix()):
        if not is_static_project_file(path):
            continue
        entries.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def main() -> int:
    entries = collect_entries()
    payload = {
        "schema_version": "1.0",
        "manifest_id": "FORMAL_PROJECT_FILE_MANIFEST_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "excludes_generated_runs": True,
        "file_count": len(entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "files": entries,
    }
    JSON_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    JSON_MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SHA_MANIFEST.write_text(
        "".join(f'{item["sha256"]} *{item["path"]}\n' for item in entries),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(JSON_MANIFEST),
                "sha_manifest": str(SHA_MANIFEST),
                "file_count": len(entries),
                "total_size_bytes": payload["total_size_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
