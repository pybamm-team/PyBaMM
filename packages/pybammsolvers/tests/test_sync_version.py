"""Tests for the version-sync pre-commit helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# pybammsolvers does not auto-assign markers from path (that lives in the pybamm
# conftest), so mark the whole module explicitly.
pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_version.py"
_spec = importlib.util.spec_from_file_location("sync_version", _SCRIPT)
sync_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_version)

_PYPROJECT_TEMPLATE = (
    "[project]\n"
    'name = "pybammsolvers"\n'
    'version = "{version}"\n'
    "\n"
    "[tool.scikit-build]\n"
    'cmake.version = ">=3.13"\n'
)
_VCPKG_TEMPLATE = (
    "{{\n"
    '  "name": "pybammsolvers",\n'
    '  "version-string": "{version}",\n'
    '  "dependencies": ["casadi"]\n'
    "}}\n"
)


class TestReadSourceVersion:
    """Parse __version__ from version.py."""

    def test_reads_version(self, tmp_path):
        version_file = tmp_path / "version.py"
        version_file.write_text('__version__ = "1.2.3"\n')
        assert sync_version.read_source_version(version_file) == "1.2.3"

    def test_raises_without_version(self, tmp_path):
        version_file = tmp_path / "version.py"
        version_file.write_text("# no version here\n")
        with pytest.raises(ValueError, match="could not find __version__"):
            sync_version.read_source_version(version_file)


class TestSyncPyproject:
    """Mirror the version into pyproject.toml's static [project] version."""

    def _pyproject(self, tmp_path, value):
        path = tmp_path / "pyproject.toml"
        path.write_text(_PYPROJECT_TEMPLATE.format(version=value))
        return path

    def test_rewrites_when_drifted(self, tmp_path):
        path = self._pyproject(tmp_path, "0.0.0")
        changed = sync_version.sync_pyproject(path, "1.2.3")
        assert changed is True
        text = path.read_text()
        assert 'version = "1.2.3"' in text
        # dependency/build version specifiers must be left untouched
        assert 'cmake.version = ">=3.13"' in text

    def test_noop_when_in_sync(self, tmp_path):
        path = self._pyproject(tmp_path, "1.2.3")
        before = path.read_text()
        changed = sync_version.sync_pyproject(path, "1.2.3")
        assert changed is False
        assert path.read_text() == before

    def test_check_reports_drift_without_writing(self, tmp_path):
        path = self._pyproject(tmp_path, "0.0.0")
        before = path.read_text()
        drifted = sync_version.sync_pyproject(path, "1.2.3", check=True)
        assert drifted is True
        assert path.read_text() == before


class TestSyncVcpkg:
    """Mirror the version into vcpkg.json's version-string."""

    def _vcpkg(self, tmp_path, value):
        path = tmp_path / "vcpkg.json"
        path.write_text(_VCPKG_TEMPLATE.format(version=value))
        return path

    def test_rewrites_when_drifted(self, tmp_path):
        path = self._vcpkg(tmp_path, "0.0.0")
        changed = sync_version.sync_vcpkg(path, "1.2.3")
        assert changed is True
        text = path.read_text()
        assert '"version-string": "1.2.3"' in text
        # surrounding manifest keys must be preserved
        assert '"dependencies": ["casadi"]' in text

    def test_noop_when_in_sync(self, tmp_path):
        path = self._vcpkg(tmp_path, "1.2.3")
        before = path.read_text()
        changed = sync_version.sync_vcpkg(path, "1.2.3")
        assert changed is False
        assert path.read_text() == before


class TestSyncGuards:
    """Syncing requires exactly one version declaration in the target."""

    def test_raises_without_declaration(self, tmp_path):
        path = tmp_path / "pyproject.toml"
        path.write_text('[project]\nname = "pybammsolvers"\n')
        with pytest.raises(ValueError, match="expected exactly one"):
            sync_version.sync_pyproject(path, "1.2.3")
