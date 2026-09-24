"""Gate optional integration dependencies without hiding broken installations."""

import json
import os
from importlib import import_module, util
from pathlib import Path

import pytest

from copper_mvp.common import PROJECT_ROOT


def pytest_addoption(parser):
    parser.addoption(
        "--require-optional",
        action="store_true",
        help="Fail instead of skipping when an optional integration runtime is missing",
    )


def missing_optional(request, reason):
    if request.config.getoption("--require-optional"):
        pytest.fail(reason)
    pytest.skip(reason)


def require_optional_module(request, name):
    if util.find_spec(name) is None:
        missing_optional(request, f"Optional module {name} is missing; install its project runtime")
    # Missing transitive dependencies, DLL failures and incompatible versions must fail.
    return import_module(name)


def activate_path(monkeypatch, variable, folder):
    directory = Path(os.environ.get(variable, PROJECT_ROOT / "runs/dependencies" / folder))
    if directory.is_dir():
        monkeypatch.syspath_prepend(str(directory))


@pytest.fixture
def sympy_module(request, monkeypatch):
    from copper_mvp.symbolic_runtime import symbolic_runtime_directory

    directory = symbolic_runtime_directory()
    if directory.is_dir():
        monkeypatch.syspath_prepend(str(directory))
    return require_optional_module(request, "sympy")


@pytest.fixture(scope="module")
def knowledge_runtime(request):
    from copper_mvp.knowledge_embedding import LocalEmbedding, load_dependencies

    load_dependencies()
    for name in ("onnxruntime", "tokenizers"):
        require_optional_module(request, name)
    model = LocalEmbedding()
    config = PROJECT_ROOT / "configs/runtime/knowledge_embedding.json"
    if not config.is_file():
        missing_optional(request, "Pinned knowledge model manifest is missing; configure the local runtime")
    manifest = json.loads(config.read_text(encoding="utf-8"))
    for name in manifest["files"]:
        if not (model.model_dir / name).is_file():
            missing_optional(request, f"Optional embedding asset {name} is missing; run scripts/setup_knowledge.py")
    # Existing but invalid assets are not an absent optional dependency.
    model._load()


@pytest.fixture
def specialized_runtime(request, monkeypatch):
    activate_path(monkeypatch, "COPPER_SPECIALIZED_RUNTIME_DIR", "specialized-models-v1")
    activate_path(monkeypatch, "COPPER_CUBIST_RUNTIME_DIR", "cubist-1.2.2")
    for name in ("catboost", "ngboost", "interpret", "cubist"):
        require_optional_module(request, name)
