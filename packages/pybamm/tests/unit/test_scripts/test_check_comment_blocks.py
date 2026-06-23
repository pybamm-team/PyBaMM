from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[5]
SCRIPT_PATH = ROOT_DIR / "scripts" / "check_comment_blocks.py"


class TestCheckCommentBlocks:
    def _run_checker(self, path):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_rejects_three_line_comment_block(self, tmp_path):
        test_file = tmp_path / "bad_comments.py"
        test_file.write_text(
            "def f():\n"
            "    # first line\n"
            "    # second line\n"
            "    # third line\n"
            "    return 1\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 1
        assert f"{test_file}:2" in result.stderr

    def test_allows_two_line_comment_block(self, tmp_path):
        test_file = tmp_path / "good_comments.py"
        test_file.write_text(
            "def f():\n    # first line\n    # second line\n    return 1\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 0
        assert result.stderr == ""

    def test_blank_line_splits_comment_blocks(self, tmp_path):
        test_file = tmp_path / "split_comments.py"
        test_file.write_text(
            "def f():\n"
            "    # first line\n"
            "    # second line\n"
            "\n"
            "    # third line\n"
            "    # fourth line\n"
            "    return 1\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 0
        assert result.stderr == ""

    def test_ignores_shebang_and_encoding_comments(self, tmp_path):
        test_file = tmp_path / "header_comments.py"
        test_file.write_text(
            "#!/usr/bin/env python3\n"
            "# -*- coding: utf-8 -*-\n"
            "def f():\n"
            "    # first line\n"
            "    # second line\n"
            "    return 1\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 0
        assert result.stderr == ""

    def test_does_not_crash_on_unparseable_python(self, tmp_path):
        test_file = tmp_path / "broken.py"
        test_file.write_text('x = "unterminated\n')

        result = self._run_checker(test_file)

        assert result.returncode == 0
        assert "Traceback" not in result.stderr

    def test_rejects_three_line_comment_block_in_yaml(self, tmp_path):
        # Apostrophe makes the file invalid as Python (unterminated string),
        # so this can only pass with a non-Python comment scanner.
        test_file = tmp_path / "workflow.yaml"
        test_file.write_text(
            "name: It's a build step\n# first line\n# second line\n# third line\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 1
        assert f"{test_file}:2" in result.stderr

    def test_allows_two_line_comment_block_in_dockerfile(self, tmp_path):
        test_file = tmp_path / "Dockerfile"
        test_file.write_text(
            "FROM python:3.12\n"
            "# install project dependencies\n"
            "# pinned via requirements.txt\n"
            "RUN echo 'don't stop'\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 0
        assert result.stderr == ""
