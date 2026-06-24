from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[5]
SCRIPT_PATH = ROOT_DIR / "scripts" / "check_notebook_output_leaks.py"


class TestCheckNotebookOutputLeaks:
    def _run_checker(self, path, *extra_args):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *extra_args, str(path)],
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

    def test_rejects_path_after_ansi_colour_code(self, tmp_path):
        # IPython prints the absolute filename directly after a colour code, so
        # "/Users/" follows the SGR terminator "m" with no separating character.
        notebook = tmp_path / "ansi.ipynb"
        self._write_notebook(
            notebook,
            {
                "ename": "E",
                "evalue": "x",
                "output_type": "error",
                "traceback": ["\x1b[1;32m/Users/alice/repo/quick_plot.py\x1b[0m\n"],
            },
        )

        result = self._run_checker(notebook)

        assert result.returncode == 1
        assert "/Users/" in result.stderr


class TestFixNotebookOutputLeaks:
    def _run_fixer(self, path):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--fix", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    def _run_checker(self, path):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    def _write_notebook(
        self, path, output, source=("print('x')\n",), ensure_ascii=True
    ):
        path.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": 1,
                            "metadata": {},
                            "outputs": [output],
                            "source": list(source),
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                },
                ensure_ascii=ensure_ascii,
            ),
            encoding="utf-8",
        )

    def test_fix_redacts_non_ascii_path_escaped(self, tmp_path):
        # Tools that escape non-ASCII store the path as \uXXXX; the fixer must
        # match that form, not only the raw multi-byte string.
        notebook = tmp_path / "nonascii_escaped.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["/Users/josé/project/café.py loaded\n"],
            },
        )

        self._run_fixer(notebook)

        value = json.loads(notebook.read_text())["cells"][0]["outputs"][0]["text"][0]
        assert value == "<path>/café.py loaded\n"
        assert self._run_checker(notebook).returncode == 0

    def test_fix_redacts_non_ascii_path_unescaped(self, tmp_path):
        # nbformat serializes notebooks as raw UTF-8; the fixer must match the
        # multi-byte path in that form too.
        notebook = tmp_path / "nonascii_raw.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["/Users/josé/project/café.py loaded\n"],
            },
            ensure_ascii=False,
        )

        self._run_fixer(notebook)

        value = json.loads(notebook.read_text())["cells"][0]["outputs"][0]["text"][0]
        assert value == "<path>/café.py loaded\n"
        assert self._run_checker(notebook).returncode == 0

    def test_fix_redacts_users_path_keeping_basename(self, tmp_path):
        notebook = tmp_path / "users.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["/Users/alice/project/file.py:1: warning\n"],
            },
        )

        result = self._run_fixer(notebook)

        text = notebook.read_text()
        assert "/Users/" not in text
        assert "<path>/file.py:1: warning" in text
        assert result.returncode == 1
        assert self._run_checker(notebook).returncode == 0

    def test_fix_redacts_traceback_file_path_keeping_basename(self, tmp_path):
        notebook = tmp_path / "traceback.ipynb"
        self._write_notebook(
            notebook,
            {
                "ename": "ValueError",
                "evalue": "bad",
                "output_type": "error",
                "traceback": [
                    'File "/home/bob/PyBaMM/src/pybamm/simulation.py", '
                    "line 472, in solve\n"
                ],
            },
        )

        self._run_fixer(notebook)

        assert "/home/" not in notebook.read_text()
        value = json.loads(notebook.read_text())["cells"][0]["outputs"][0]["traceback"][
            0
        ]
        assert value == 'File "<path>/simulation.py", line 472, in solve\n'

    def test_fix_redacts_vscode_uri_to_placeholder(self, tmp_path):
        notebook = tmp_path / "vscode.ipynb"
        self._write_notebook(
            notebook,
            {
                "ename": "AttributeError",
                "evalue": "bad",
                "output_type": "error",
                "traceback": [
                    "<a href='vscode-notebook-cell:/Users/alice/repo/demo.ipynb"
                    "#W6sZmlsZQ%3D%3D?line=5'>6</a>\n"
                ],
            },
        )

        self._run_fixer(notebook)

        text = notebook.read_text()
        assert "vscode-notebook-cell:" not in text
        assert "/Users/" not in text
        assert "<a href='<path>'>6</a>" in text
        assert self._run_checker(notebook).returncode == 0

    def test_fix_redacts_windows_path_keeping_basename(self, tmp_path):
        notebook = tmp_path / "windows.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["C:\\Users\\bob\\repo\\model.py loaded\n"],
            },
        )

        self._run_fixer(notebook)

        text = json.loads(notebook.read_text())["cells"][0]["outputs"][0]["text"][0]
        assert "C:\\Users\\" not in text
        assert text == "<path>\\model.py loaded\n"
        assert self._run_checker(notebook).returncode == 0

    def test_fix_redacts_posixpath_data_field(self, tmp_path):
        notebook = tmp_path / "data.ipynb"
        self._write_notebook(
            notebook,
            {
                "data": {"text/plain": ["PosixPath('/home/user/.cache/Ecker_1C.csv')"]},
                "execution_count": 1,
                "metadata": {},
                "output_type": "execute_result",
            },
        )

        self._run_fixer(notebook)

        value = json.loads(notebook.read_text())["cells"][0]["outputs"][0]["data"][
            "text/plain"
        ][0]
        assert value == "PosixPath('<path>/Ecker_1C.csv')"

    def test_fix_redacts_path_after_ansi_colour_code(self, tmp_path):
        notebook = tmp_path / "ansi_path.ipynb"
        self._write_notebook(
            notebook,
            {
                "ename": "E",
                "evalue": "x",
                "output_type": "error",
                "traceback": ["\x1b[1;32m/Users/alice/repo/quick_plot.py\x1b[0m\n"],
            },
        )

        self._run_fixer(notebook)

        value = json.loads(notebook.read_text())["cells"][0]["outputs"][0]["traceback"][
            0
        ]
        assert value == "\x1b[1;32m<path>/quick_plot.py\x1b[0m\n"
        assert self._run_checker(notebook).returncode == 0

    def test_fix_leaves_clean_notebook_byte_identical(self, tmp_path):
        notebook = tmp_path / "clean.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["Voltage [V]: 4.2\n"],
            },
        )
        before = notebook.read_bytes()

        result = self._run_fixer(notebook)

        assert result.returncode == 0
        assert notebook.read_bytes() == before

    def test_fix_preserves_unrelated_escape_casing(self, tmp_path):
        # Mixes uppercase ANSI escapes with a leaked path: the fix must touch only
        # the path span, since Python's json would rewrite \x1b to lowercase.
        notebook = tmp_path / "ansi.ipynb"
        raw = (
            '{\n "cells": [\n  {\n   "cell_type": "code",\n'
            '   "execution_count": 1,\n   "metadata": {},\n   "outputs": [\n    {\n'
            '     "output_type": "error",\n     "ename": "E",\n     "evalue": "x",\n'
            '     "traceback": [\n'
            '      "\\u001B[0;32mFile \\"/Users/alice/repo/quick_plot.py\\", '
            'line 146\\u001B[0m\\n",\n'
            '      "\\u001B[0;31mAttributeError\\u001B[0m: boom"\n'
            "     ]\n    }\n   ],\n"
            '   "source": [\n    "print(1)"\n   ]\n  }\n ],\n'
            '  "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}\n'
        )
        notebook.write_text(raw)

        self._run_fixer(notebook)

        text = notebook.read_text()
        assert "/Users/" not in text
        assert "\\u001B[0;31mAttributeError\\u001B[0m: boom" in text
        assert (
            '\\u001B[0;32mFile \\"<path>/quick_plot.py\\", line 146\\u001B[0m' in text
        )
        assert self._run_checker(notebook).returncode == 0

    def test_fix_does_not_touch_paths_only_in_source(self, tmp_path):
        notebook = tmp_path / "source.ipynb"
        self._write_notebook(
            notebook,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["all good\n"],
            },
            source=["model = pybamm.load('/home/example/model.json')\n"],
        )
        before = notebook.read_bytes()

        result = self._run_fixer(notebook)

        assert result.returncode == 0
        assert notebook.read_bytes() == before
