"""只读验收正式工程的路径、哈希、模型、数据和排除边界。"""

from __future__ import annotations

import hashlib
import json
import argparse
import os
from pathlib import Path
import re
import sys
import tomllib
from typing import Any
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_file_manifest import collect_entries, sha256_file  # noqa: E402


REQUIRED_PATHS = (
    ".env.example",
    "README.md",
    "pyproject.toml",
    "requirements-verified.txt",
    "configs/agents/graph_v1.yaml",
    "configs/contracts/core_features_v2.yaml",
    "configs/contracts/timing_contract_v2.yaml",
    "configs/llm/providers.yaml",
    "contracts/frozen/external_runtime_freeze_manifest_v1.json",
    "contracts/frozen/selected_model_manifest_v1.json",
    "data/development_2024_2025/core_feature_matrix_v2.csv",
    "data/development_2024_2025/as_of_admission_card_core_v2.csv",
    "data/development_2024_2025/training_evaluation_index_v2.csv",
    "data/development_2024_2025/cv_fold_manifest_v2.csv",
    "data/development_2024_2025/p2_run_manifest_v1.json",
    "models/selected_model_manifest_v1.json",
    "scripts/run_langgraph_synthetic_v1.py",
    "scripts/run_langgraph_oof_parity_v1.py",
    "scripts/run_v1_offline_smoke_v1.py",
    "src/copper_langgraph_v2/graph.py",
    "src/copper_langgraph_v2/paths.py",
    "src/copper_mas/models/predictors.py",
    "tests/v2/test_langgraph_parity.py",
    "provenance/frozen_input_hashes_v1.json",
    "provenance/file_manifest_v1.json",
    "MANIFEST.sha256",
    "requirements-mvp-lock.txt",
    "scripts/run_mvp.py",
    "src/copper_mvp/api.py",
    "src/copper_mvp/diagnostic_agent.py",
    "src/copper_mvp/diagnostic_tools.py",
    "configs/llm/diagnostic_agent.yaml",
    "web/package.json",
    "web/src/main.tsx",
    "web/src/App.tsx",
    "web/src/AgentDiagnosisPanel.tsx",
    "web/src/ExplanationPanel.tsx",
    "src/copper_mvp/data_contracts.py",
    "src/copper_mvp/data_service.py",
    "src/copper_mvp/labels.py",
    "src/copper_mvp/api_data.py",
    "scripts/verify_g1.py",
    "src/copper_mvp/model_registry.py",
    "src/copper_mvp/model_adapters.py",
    "src/copper_mvp/model_training.py",
    "src/copper_mvp/model_evaluation.py",
    "src/copper_mvp/model_comparisons.py",
    "src/copper_mvp/api_models.py",
    "scripts/run_model_comparison.py",
    "src/copper_mvp/optimization_problem.py",
    "src/copper_mvp/optimizer_registry.py",
    "src/copper_mvp/optimizer_comparison.py",
    "src/copper_mvp/api_optimizers.py",
    "scripts/run_optimizer_comparison.py",
    "src/copper_mvp/access.py",
    "src/copper_mvp/research_store.py",
    "src/copper_mvp/research_tools.py",
    "src/copper_mvp/research_service.py",
    "src/copper_mvp/api_research.py",
    "configs/llm/research_agent.yaml",
    "web/src/AccessGate.tsx",
    "web/src/ResearchPanel.tsx",
    "scripts/verify_research_live.py",
)

FORBIDDEN_DIR_NAMES = {
    ".pytest_cache",
    ".pytest_tmp",
    ".venv",
    "__pycache__",
    "node_modules",
    "previews",
}
FORBIDDEN_FILE_SUFFIXES = {".db", ".joblib", ".pyc", ".sqlite", ".sqlite3"}
FORBIDDEN_DATA_TOKENS = {"2026", "external", "outcome_ledger", "sealed"}


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))


def _check_required(errors: list[str], checks: list[str]) -> None:
    missing = [relative for relative in REQUIRED_PATHS if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        errors.append(f"缺少必需文件: {missing}")
    else:
        checks.append(f"必需文件 {len(REQUIRED_PATHS)}/{len(REQUIRED_PATHS)} 存在")


def _check_exclusions(errors: list[str], checks: list[str], *, distribution: bool = False) -> None:
    forbidden: list[str] = []
    runtime_names = FORBIDDEN_DIR_NAMES | {"runs", "dist", "build", ".mypy_cache", ".ruff_cache"}
    for directory, subdirs, filenames in os.walk(PROJECT_ROOT):
        keep = []
        for name in subdirs:
            relative = (Path(directory) / name).relative_to(PROJECT_ROOT)
            if name == ".git" or name.endswith(".egg-info"):
                continue
            if name in runtime_names or name.startswith((".venv", ".pytest_tmp")):
                if distribution:
                    forbidden.append(relative.as_posix() + "/")
            else:
                keep.append(name)
        subdirs[:] = keep
        for name in filenames:
            path = Path(directory) / name
            relative = path.relative_to(PROJECT_ROOT)
            if name != ".env.example" and (name == ".env" or name.startswith(".env.")):
                if distribution:
                    forbidden.append(relative.as_posix())
                continue  # Private configuration is never read.
            if path.suffix.lower() in FORBIDDEN_FILE_SUFFIXES:
                forbidden.append(relative.as_posix())
            if relative.parts and relative.parts[0] == "data" and any(token in relative.as_posix().lower() for token in FORBIDDEN_DATA_TOKENS):
                forbidden.append(relative.as_posix())
    if forbidden:
        errors.append(f"发现应排除的环境、缓存、模型或外部数据文件: {sorted(set(forbidden))}")
    else:
        checks.append("分发包排除规则匹配" if distribution else "开发目录检查通过；运行目录、环境与构建缓存已跳过，正式数据无 2026/真值账本")


def _check_frozen_inputs(errors: list[str], checks: list[str]) -> None:
    payload = _load_json("provenance/frozen_input_hashes_v1.json")
    mismatch: list[str] = []
    for relative, expected in payload["files"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            mismatch.append(f"{relative}: missing")
            continue
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != expected["sha256"] or actual_size != expected["size_bytes"]:
            mismatch.append(
                f"{relative}: sha256={actual_hash}, size={actual_size}"
            )
    if mismatch:
        errors.append(f"冻结输入哈希或大小不一致: {mismatch}")
    else:
        checks.append(f"冻结输入 {len(payload['files'])}/{len(payload['files'])} 哈希匹配")

    expected_freeze_hash = payload["original_v1_freeze"]["manifest_sha256"]
    freeze_path = PROJECT_ROOT / "contracts/frozen/external_runtime_freeze_manifest_v1.json"
    if freeze_path.is_file() and sha256_file(freeze_path) != expected_freeze_hash:
        errors.append("V1 冻结运行时清单自身 SHA-256 不匹配")
    else:
        checks.append("V1 冻结运行时清单自身哈希匹配")


def _check_p2_and_model(errors: list[str], checks: list[str]) -> None:
    data_dir = PROJECT_ROOT / "data/development_2024_2025"
    p2 = _load_json("data/development_2024_2025/p2_run_manifest_v1.json")
    p2_expected = p2.get("source_sha256") or {}
    mismatch = []
    for filename in (
        "core_feature_matrix_v2.csv",
        "training_evaluation_index_v2.csv",
        "cv_fold_manifest_v2.csv",
    ):
        actual = sha256_file(data_dir / filename)
        if p2_expected.get(filename) != actual:
            mismatch.append(filename)
    if (p2.get("safety") or {}).get("external_2026_read") is not False:
        mismatch.append("P2 external_2026_read")
    if mismatch:
        errors.append(f"P2 来源或安全声明不一致: {mismatch}")
    else:
        checks.append("P2 三个受清单约束的数据哈希与 external_2026_read=false 匹配")

    model_path = PROJECT_ROOT / "models/selected_model_manifest_v1.json"
    contract_copy = PROJECT_ROOT / "contracts/frozen/selected_model_manifest_v1.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    predictor = model.get("a4_predictor") or {}
    predictor_path = PROJECT_ROOT / "src/copper_mas/models/predictors.py"
    model_errors = []
    if sha256_file(model_path) != sha256_file(contract_copy):
        model_errors.append("模型清单的 models/ 与 contracts/frozen/ 副本不同")
    if predictor.get("model_id") != "PERSISTENCE_CURRENT_RESULT_V1":
        model_errors.append("model_id")
    if predictor.get("serialized_model") is not False:
        model_errors.append("serialized_model")
    if model.get("external_2026_read") is not False:
        model_errors.append("external_2026_read")
    expected_code = (model.get("source_sha256") or {}).get("persistence_predictor_code")
    if expected_code != sha256_file(predictor_path):
        model_errors.append("PersistencePredictor 代码哈希")
    if model_errors:
        errors.append(f"冻结模型检查失败: {model_errors}")
    else:
        checks.append("Persistence 模型 ID、无序列化声明和预测器代码哈希匹配")


def _check_v1_python_freeze(errors: list[str], checks: list[str]) -> None:
    freeze = _load_json("contracts/frozen/external_runtime_freeze_manifest_v1.json")
    source_entries = [item for item in freeze["files"] if item["path"].startswith("src/")]
    mismatch = []
    for item in source_entries:
        path = PROJECT_ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            mismatch.append(item["path"])
    if len(source_entries) != 37:
        mismatch.append(f"冻结清单 src 文件数={len(source_entries)}，预期=37")
    if mismatch:
        errors.append(f"原样复制的 V1 Python 冻结文件不一致: {mismatch}")
    else:
        checks.append("V1 Python 冻结文件 37/37 哈希匹配")


def _check_file_manifest(errors: list[str], checks: list[str]) -> None:
    manifest = _load_json("provenance/file_manifest_v1.json")
    expected = {item["path"]: item for item in manifest.get("files", [])}
    actual_entries = collect_entries()
    actual = {item["path"]: item for item in actual_entries}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatch = sorted(
        path
        for path in set(expected) & set(actual)
        if expected[path]["sha256"] != actual[path]["sha256"]
        or expected[path]["size_bytes"] != actual[path]["size_bytes"]
    )
    if missing or extra or mismatch:
        errors.append(
            f"全目录清单不一致: missing={missing}, extra={extra}, mismatch={mismatch}"
        )
        return
    sha_text = "".join(
        f'{item["sha256"]} *{item["path"]}\n' for item in actual_entries
    )
    stored_sha_text = (PROJECT_ROOT / "MANIFEST.sha256").read_text(encoding="utf-8")
    if sha_text != stored_sha_text:
        errors.append("MANIFEST.sha256 与 JSON 清单不一致")
    else:
        checks.append(f"全目录静态文件 {len(actual_entries)}/{len(actual_entries)} 哈希匹配")


def _check_imports(errors: list[str], checks: list[str]) -> None:
    try:
        import copper_mas  # noqa: F401
        from copper_langgraph_v2.graph import LangGraphRunner  # noqa: F401
        from copper_mas.models.predictors import PersistencePredictor  # noqa: F401
        from copper_mvp.api import create_app
        from copper_mvp.common import APP_VERSION
        from copper_mvp.diagnostic_agent import AgentSettings
        AgentSettings.load()
        project_version = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        if APP_VERSION != project_version or create_app().version != APP_VERSION:
            raise ValueError("MVP API版本与pyproject.toml不一致")
    except Exception as exc:  # pragma: no cover - 仅作为安装验收
        errors.append(f"关键包导入失败: {type(exc).__name__}: {exc}")
    else:
        checks.append("V1/V2与MVP关键入口导入成功；诊断配置和应用版本一致")


def _check_document_links(errors: list[str], checks: list[str]) -> None:
    """Check current local Markdown links without rewriting archived originals."""
    candidates = [PROJECT_ROOT / name for name in ("README.md", "文件清单.md", "data/README.md", "models/README.md", "scripts/README.md", "provenance/README.md")]
    candidates.extend((PROJECT_ROOT / "docs").rglob("*.md"))
    broken = []
    checked = 0
    for path in sorted(set(candidates)):
        if not path.is_file() or "legacy" in path.relative_to(PROJECT_ROOT).parts:
            continue
        in_code = False
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith(("```", "~~~")):
                in_code = not in_code
                continue
            if in_code:
                continue
            for match in re.finditer(r"\[[^\]\n]*\]\(([^)\n]+)\)", line):
                raw = match.group(1).strip()
                if not raw:
                    continue
                target = raw[1:raw.index(">")] if raw.startswith("<") and ">" in raw else raw.split()[0]
                if target.startswith("#"):
                    continue
                if urlsplit(target).scheme and not re.match(r"^[A-Za-z]:[\\/]", target):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not target:
                    continue
                checked += 1
                if not (path.parent / target).exists():
                    broken.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:{line_number} -> {target}")
    if broken:
        errors.append(f"当前文档存在失效本地链接: {broken}")
    else:
        checks.append(f"当前文档本地链接 {checked}/{checked} 有效；历史原稿未改写")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", action="store_true", help="检查干净分发包；默认检查开发工作区")
    args = parser.parse_args()
    errors: list[str] = []
    checks: list[str] = []
    _check_required(errors, checks)
    if not errors:
        _check_exclusions(errors, checks, distribution=args.distribution)
        _check_frozen_inputs(errors, checks)
        _check_p2_and_model(errors, checks)
        _check_v1_python_freeze(errors, checks)
        _check_file_manifest(errors, checks)
        _check_imports(errors, checks)
        _check_document_links(errors, checks)
    report = {
        "status": "PASSED" if not errors else "FAILED",
        "project_root": str(PROJECT_ROOT),
        "checks": checks,
        "errors": errors,
        "external_2026_read": False,
        "network_calls_made": 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
