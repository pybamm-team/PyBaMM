import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[4] / "scripts" / "check_changelog_fragments.py"
spec = importlib.util.spec_from_file_location("check_changelog_fragments", SCRIPT)
check_changelog_fragments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_changelog_fragments)


@pytest.fixture
def fragment_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(check_changelog_fragments, "REPO_ROOT", tmp_path)
    directory = tmp_path / "changelog.d"
    directory.mkdir()
    (directory / "README.md").write_text("# Changelog fragments\n")
    return directory


def set_changes(monkeypatch, *lines):
    monkeypatch.setattr(
        check_changelog_fragments, "changed_paths", lambda base: list(lines)
    )


class TestCheckChangelogFragments:
    def test_accepts_valid_fragments(self, fragment_dir):
        (fragment_dir / "1234.bugfix.md").write_text("Fixed a thing.\n")
        (fragment_dir / "1234.bugfix.1.md").write_text("Fixed another.\n- Detail.\n")

        assert check_changelog_fragments.main([]) == 0

    @pytest.mark.parametrize(
        ("name", "text", "message"),
        [
            ("1234.fix.md", "Fixed.", "name it"),
            ("bugfix.md", "Fixed.", "name it"),
            ("1234.bugfix.md", "  \n", "empty"),
            ("1234.bugfix.md", "- Fixed.", "leading bullet"),
            (
                "1234.bugfix.md",
                "Fixed. ([#1234](https://github.com/pybamm-team/PyBaMM/pull/1234))",
                "PR link",
            ),
        ],
    )
    def test_reports_invalid_fragments(self, fragment_dir, capsys, name, text, message):
        (fragment_dir / name).write_text(text)

        assert check_changelog_fragments.main([]) == 1
        assert message in capsys.readouterr().err

    def test_reports_added_fragment_named_after_another_pr(
        self, fragment_dir, monkeypatch, capsys
    ):
        (fragment_dir / "1233.feature.md").write_text("Added a thing.\n")
        set_changes(monkeypatch, "A\tchangelog.d/1233.feature.md")

        assert check_changelog_fragments.main(["--pr", "1234"]) == 1
        assert "this PR's number, 1234" in capsys.readouterr().err

    def test_ignores_fragments_the_pr_does_not_add(self, fragment_dir, monkeypatch):
        (fragment_dir / "1200.feature.md").write_text("Added a thing.\n")
        (fragment_dir / "1234.feature.md").write_text("Added another.\n")
        set_changes(
            monkeypatch,
            "M\tchangelog.d/1200.feature.md",
            "A\tchangelog.d/1234.feature.md",
        )

        assert check_changelog_fragments.main(["--pr", "1234"]) == 0

    def test_exempts_prs_that_edit_the_changelog(self, fragment_dir, monkeypatch):
        (fragment_dir / "1233.feature.md").write_text("Added a thing.\n")
        set_changes(monkeypatch, "A\tchangelog.d/1233.feature.md", "M\tCHANGELOG.md")

        assert check_changelog_fragments.main(["--pr", "1234"]) == 0
