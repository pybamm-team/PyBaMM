from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[5]
SCRIPT_PATH = ROOT_DIR / "scripts" / "check_changelog.py"


class TestCheckChangelog:
    def _run_checker(self, path):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_allows_valid_unreleased_entries(self, tmp_path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n\n"
            "## Features\n\n"
            "- Add new hook support "
            "([#9999](https://github.com/pybamm-team/PyBaMM/pull/9999))\n"
        )

        result = self._run_checker(changelog)

        assert result.returncode == 0

    def test_rejects_missing_pr_link(self, tmp_path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n\n"
            "## Bug fixes\n\n"
            "- Fix a regression\n"
        )

        result = self._run_checker(changelog)

        assert result.returncode == 1
        assert "must end with a PR link" in result.stderr

    def test_rejects_bullet_outside_allowed_unreleased_subsection(self, tmp_path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n\n"
            "- Loose bullet "
            "([#9999](https://github.com/pybamm-team/PyBaMM/pull/9999))\n"
        )

        result = self._run_checker(changelog)

        assert result.returncode == 1
        assert "allowed subsection" in result.stderr

    def test_rejects_breaking_change_without_migration_note(self, tmp_path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n\n"
            "## Breaking changes\n\n"
            "- ([#9999](https://github.com/pybamm-team/PyBaMM/pull/9999))\n"
        )

        result = self._run_checker(changelog)

        assert result.returncode == 1
        assert "migration note" in result.stderr
