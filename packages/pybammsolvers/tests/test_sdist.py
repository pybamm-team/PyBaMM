"""Tests for the contents of the pybammsolvers source distribution."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSdistContents:
    """Guard non-Python build inputs that source installs need but wheels don't."""

    pytestmark = pytest.mark.unit

    def test_sdist_ships_vcpkg_overlay(self, tmp_path):
        """
        The Windows vcpkg build reads ``vcpkg-configuration.json``'s ``overlay-ports``,
        so the referenced ``vcpkg-overlays/`` port must be in the sdist or a source
        install fails before the build starts.
        """
        if not (PROJECT_ROOT / "pyproject.toml").exists():
            pytest.skip("not a source checkout")
        pytest.importorskip("scikit_build_core")

        out = tmp_path / "sdist"
        out.mkdir()
        # Build from PROJECT_ROOT in a subprocess so the chdir can't leak into the
        # (possibly parallel) test session.
        script = (
            "import os;"
            f"os.chdir({str(PROJECT_ROOT)!r});"
            "from scikit_build_core.build import build_sdist;"
            f"build_sdist({str(out)!r})"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, f"sdist build failed:\n{result.stderr}"

        (archive,) = out.glob("*.tar.gz")
        with tarfile.open(archive) as tar:
            # Drop the leading "pybammsolvers-<version>/" prefix from each member.
            names = {n.split("/", 1)[1] for n in tar.getnames() if "/" in n}

        for expected in (
            "vcpkg-overlays/sundials/vcpkg.json",
            "vcpkg-overlays/sundials/portfile.cmake",
            "vcpkg-overlays/sundials/find-klu.patch",
        ):
            assert expected in names, f"{expected} missing from sdist"
