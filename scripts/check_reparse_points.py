"""Report Windows reparse points (junctions and symlinks) inside the project.

A dangling junction is invisible to os.walk (silently skipped) but makes
pathlib.rglob raise FileNotFoundError, so recursive scans behave inconsistently.
One such junction, left over from an earlier project location, is why the
documentation link audit had to stop using rglob.

Read-only: this script never creates or removes a reparse point.
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
JUNCTION_TAG = 0xA0000003


def classify(st: os.stat_result) -> str | None:
    if stat.S_ISLNK(st.st_mode):
        return "symlink"
    if not getattr(st, "st_file_attributes", 0) & REPARSE:
        return None
    tag = getattr(st, "st_reparse_tag", None)
    if tag == JUNCTION_TAG:
        return "junction"
    return f"reparse(0x{tag:x})" if tag else "reparse"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="项目根目录")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    found: list[tuple[str, str, bool, str]] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=lambda e: None):
        for name in list(dirnames) + filenames:
            path = Path(dirpath) / name
            try:
                st = os.lstat(path)
            except OSError:
                continue
            kind = classify(st)
            if kind is None:
                continue
            if name in dirnames:
                dirnames.remove(name)
            try:
                target = os.readlink(path)
            except OSError:
                target = "<不可读>"
            try:
                reachable = path.exists()
            except OSError:
                reachable = False
            found.append((path.relative_to(root).as_posix(), target, reachable, kind))

    for rel, target, reachable, kind in sorted(found):
        print(f"[{'正常' if reachable else '悬空'}] ({kind}) {rel}\n    -> {target}")
    dangling = [f for f in found if not f[2]]
    print(f"重解析点 {len(found)} 个，其中悬空 {len(dangling)} 个")
    if dangling:
        print("悬空项会使 pathlib.rglob 抛 FileNotFoundError；确认目标为空后可用 rmdir 移除。")
    return 1 if dangling else 0


if __name__ == "__main__":
    raise SystemExit(main())
