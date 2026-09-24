"""Missing optional packages may skip; installed-but-broken packages must fail."""

from types import SimpleNamespace

import conftest
import pytest

from copper_mvp.common import PROJECT_ROOT
from copper_mvp.symbolic_runtime import symbolic_runtime_directory


def request_stub(strict=False):
    return SimpleNamespace(config=SimpleNamespace(getoption=lambda name: strict))


def test_absent_optional_module_is_explicitly_skipped(monkeypatch):
    monkeypatch.setattr(conftest.util, "find_spec", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match="Optional module"):
        conftest.require_optional_module(request_stub(), "missing_fixture_package")


def test_strict_verification_rejects_missing_optional_runtime(monkeypatch):
    monkeypatch.setattr(conftest.util, "find_spec", lambda name: None)
    with pytest.raises(pytest.fail.Exception, match="Optional module"):
        conftest.require_optional_module(request_stub(True), "missing_fixture_package")


def test_installed_but_broken_optional_dependency_is_not_skipped(monkeypatch):
    monkeypatch.setattr(conftest.util, "find_spec", lambda name: object())

    def broken(name):
        raise ModuleNotFoundError("Missing transitive dependency")

    monkeypatch.setattr(conftest, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="transitive"):
        conftest.require_optional_module(request_stub(), "installed_fixture_package")


def test_symbolic_default_matches_project_runtime_without_env(monkeypatch):
    monkeypatch.delenv("COPPER_SYMBOLIC_RUNTIME_DIR", raising=False)
    assert symbolic_runtime_directory() == (PROJECT_ROOT / "runs/dependencies/pysr-1.5.9").resolve()


def test_symbolic_runtime_honors_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COPPER_SYMBOLIC_RUNTIME_DIR", str(tmp_path))
    assert symbolic_runtime_directory() == tmp_path.resolve()
