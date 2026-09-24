from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def verifier(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("maintenance_verifier", scripts / "verify_project.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing", ["src/copper_mvp/api.py", "configs/llm/diagnostic_agent.yaml"])
def test_missing_current_runtime_is_detected(verifier, tmp_path, monkeypatch, missing):
    monkeypatch.setattr(verifier, "PROJECT_ROOT", tmp_path)
    for name in verifier.REQUIRED_PATHS:
        if name != missing:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    errors, checks = [], []
    verifier._check_required(errors, checks)
    assert any(missing in error for error in errors)


def test_document_link_check_catches_live_links_and_preserves_archive(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "PROJECT_ROOT", tmp_path)
    (tmp_path / "docs/legacy").mkdir(parents=True)
    (tmp_path / "docs/ok.md").write_text("# existing", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[ok](docs/ok.md)\n[missing](docs/missing.md)\n[web](https://example.com)\n```text\n[example](not-a-file)\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/legacy/original.md").write_text("[old](historical-path.md)", encoding="utf-8")
    errors, checks = [], []
    verifier._check_document_links(errors, checks)
    assert len(errors) == 1 and "missing.md" in errors[0]
    assert "not-a-file" not in errors[0] and "historical-path.md" not in errors[0]


def test_manifest_is_idempotent_and_tracks_content_changes(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location("manifest_builder", scripts / "build_file_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "JSON_MANIFEST", tmp_path / "provenance/file_manifest_v1.json")
    monkeypatch.setattr(module, "SHA_MANIFEST", tmp_path / "MANIFEST.sha256")
    source = tmp_path / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")
    assert module.main() == 0
    before = (module.JSON_MANIFEST.read_bytes(), module.SHA_MANIFEST.read_bytes())
    assert module.main() == 0
    assert before == (module.JSON_MANIFEST.read_bytes(), module.SHA_MANIFEST.read_bytes())
    source.write_text("value = 2\n", encoding="utf-8")
    assert module.main() == 0
    assert before[1] != module.SHA_MANIFEST.read_bytes()
