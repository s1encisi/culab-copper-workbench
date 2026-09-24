"""为正式工程生成逐文件 SHA-256 清单。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import UTC, datetime
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
    # 浏览器自动化调试缓存（已被 .gitignore 排除）：内容随每次运行变化，
    # 纳入清单会使完整性校验产生无意义抖动。
    ".playwright-cli",
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
        part in EXCLUDED_DIR_NAMES or part.startswith(".venv") or part.endswith(".egg-info") for part in relative.parts
    ):
        return False
    if path.name != ".env.example" and (path.name == ".env" or path.name.startswith(".env.")):
        return False
    return path.is_file()


def is_reparse_point(path: Path) -> bool:
    status = path.lstat()
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def collect_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    files = []
    for directory, subdirs, filenames in os.walk(PROJECT_ROOT):
        subdirs[:] = [
            name
            for name in subdirs
            if name not in EXCLUDED_DIR_NAMES
            and not name.startswith((".venv", ".pytest_tmp"))
            and not name.endswith(".egg-info")
            and not is_reparse_point(Path(directory) / name)
        ]
        files.extend(Path(directory) / name for name in filenames)
    for path in sorted(files, key=lambda item: item.as_posix()):
        if is_reparse_point(path) or not is_static_project_file(path):
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
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "root": ".",
        "excludes_generated_runs": True,
        "file_count": len(entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "files": entries,
    }
    if JSON_MANIFEST.is_file():
        try:
            previous = json.loads(JSON_MANIFEST.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            previous = {}
        if (
            isinstance(previous, dict)
            and all(previous.get(key) == value for key, value in payload.items() if key != "generated_at_utc")
            and isinstance(previous.get("generated_at_utc"), str)
        ):
            payload["generated_at_utc"] = previous["generated_at_utc"]
    JSON_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    JSON_MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SHA_MANIFEST.write_text(
        "".join(f"{item['sha256']} *{item['path']}\n" for item in entries),
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
