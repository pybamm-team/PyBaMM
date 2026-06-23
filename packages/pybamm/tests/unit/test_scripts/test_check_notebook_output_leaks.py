from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[5]
SCRIPT_PATH = ROOT_DIR / "scripts" / "check_notebook_output_leaks.py"


class TestCheckNotebookOutputLeaks:
    def _run_checker(self, path):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    def _write_notebook(self, path, output):
        path.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": 1,
                            "metadata": {},
                            "outputs": [output],
                            "source": ["print('x')\n"],
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            )
        )

    def test_rejects_users_path_in_stream_output(self, tmp_path):
        notebook = tmp_path / "users.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["/Users/alice/project/file.py:1: warning\n"],
            },
        )

        result = self._run_checker(notebook)

        assert result.returncode == 1
        assert "cell 1, output 1" in result.stderr
        assert "/Users/" in result.stderr

    def test_rejects_home_path_in_execute_result(self, tmp_path):
        notebook = tmp_path / "home.ipynb"
        self._write_notebook(
            notebook,
            {
                "data": {"text/plain": ["PosixPath('/home/user/.cache/x')"]},
                "execution_count": 1,
                "metadata": {},
                "output_type": "execute_result",
            },
        )

        result = self._run_checker(notebook)

        assert result.returncode == 1
        assert "data:text/plain" in result.stderr
        assert "/home/" in result.stderr

    def test_rejects_vscode_notebook_traceback_uri(self, tmp_path):
        notebook = tmp_path / "traceback.ipynb"
        self._write_notebook(
            notebook,
            {
                "ename": "AttributeError",
                "evalue": "bad",
                "output_type": "error",
                "traceback": [
                    "vscode-notebook-cell:/Users/alice/project/demo.ipynb#X\n"
                ],
            },
        )

        result = self._run_checker(notebook)

        assert result.returncode == 1
        assert "traceback" in result.stderr
        assert "vscode-notebook-cell:" in result.stderr

    def test_allows_clean_notebook_outputs(self, tmp_path):
        notebook = tmp_path / "clean.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["Voltage [V]: 4.2\n"],
            },
        )

        result = self._run_checker(notebook)

        assert result.returncode == 0
        assert result.stderr == ""

    def test_rejects_path_in_error_evalue(self, tmp_path):
        notebook = tmp_path / "evalue.ipynb"
        self._write_notebook(
            notebook,
            {
                "ename": "FileNotFoundError",
                "evalue": "/Users/alice/secret.py not found",
                "output_type": "error",
                "traceback": ["clean traceback line\n"],
            },
        )

        result = self._run_checker(notebook)

        assert result.returncode == 1
        assert "evalue" in result.stderr
        assert "/Users/" in result.stderr

    def test_allows_home_substring_inside_url(self, tmp_path):
        notebook = tmp_path / "url.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["See https://docs.example.com/home/intro\n"],
            },
        )

        result = self._run_checker(notebook)

        assert result.returncode == 0
        assert result.stderr == ""
