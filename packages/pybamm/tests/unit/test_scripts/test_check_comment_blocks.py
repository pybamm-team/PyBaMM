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

    def test_blank_lines_do_not_split_comment_blocks(self, tmp_path):
        # Blank lines with no code between are still one logical block: this is
        # the loophole (a long comment chopped into "compliant" chunks) closed.
        test_file = tmp_path / "blank_split.py"
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

        assert result.returncode == 1
        assert f"{test_file}:2" in result.stderr

    def test_code_between_splits_comment_blocks(self, tmp_path):
        # Real code between the two 2-line blocks keeps them independent.
        test_file = tmp_path / "code_split.py"
        test_file.write_text(
            "def f():\n"
            "    # first line\n"
            "    # second line\n"
            "    x = 1\n"
            "    # third line\n"
            "    # fourth line\n"
            "    return x\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 0
        assert result.stderr == ""

    def test_allows_two_lines_split_by_blank(self, tmp_path):
        # Merging across blanks does not over-flag: two lines total is still fine.
        test_file = tmp_path / "two_split.py"
        test_file.write_text(
            "def f():\n    # first line\n\n    # second line\n    return 1\n"
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

    def test_allows_license_header_block(self, tmp_path):
        test_file = tmp_path / "licensed.py"
        test_file.write_text(
            "# Copyright 2018 Google LLC\n"
            "#\n"
            '# Licensed under the Apache License, Version 2.0 (the "License");\n'
            "# you may not use this file except in compliance with the License.\n"
            "import pybamm\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 0
        assert result.stderr == ""

    def test_rejects_long_block_without_license_marker(self, tmp_path):
        test_file = tmp_path / "no_marker.py"
        test_file.write_text(
            "# first line\n# second line\n# third line\nimport pybamm\n"
        )

        result = self._run_checker(test_file)

        assert result.returncode == 1
        assert f"{test_file}:1" in result.stderr

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
