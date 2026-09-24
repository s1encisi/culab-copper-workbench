"""Verify that every intra-project import name actually exists in its module.

py_compile only validates syntax, and neither F401 nor F821 catches
"imported a name that the target module does not define". That class of defect
therefore compiles cleanly and only fails at runtime; a stale import in
diagnostic_tools once broke scripts/demo_public.py at its first import and
prevented 15 test modules from being collected.

This check parses module-level bindings statically and never executes project
code.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

PACKAGE_PREFIX = "copper_"


def bindings(path: Path) -> set[str]:
    """Collect every name a module exposes at top level."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add("*" if alias.name == "*" else alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            # Conditional imports count as bindings.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in sub.names if a.name != "*")
                elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
    return names


def locate(src: Path, module: str) -> Path | None:
    rel = module.replace(".", "/")
    for candidate in (src / f"{rel}.py", src / rel / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="项目根目录")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    src = root / "src"
    cache: dict[Path, set[str]] = {}
    problems: list[str] = []
    checked = 0

    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(PACKAGE_PREFIX):
                continue
            target = locate(src, node.module)
            if target is None:
                problems.append(f"{path.relative_to(root)}:{node.lineno} 模块不存在: {node.module}")
                continue
            if target not in cache:
                cache[target] = bindings(target)
            if "*" in cache[target]:
                continue
            for alias in node.names:
                checked += 1
                if alias.name not in cache[target]:
                    problems.append(f"{path.relative_to(root)}:{node.lineno} `{node.module}` 未定义 `{alias.name}`")

    print(f"内部导入名 {checked} 个，目标模块 {len(cache)} 个")
    for problem in problems:
        print(f"  ! {problem}")
    print(f"不可解析: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
