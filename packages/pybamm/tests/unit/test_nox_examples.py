from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

NOXFILE = Path(__file__).parents[4] / "noxfile.py"


class RecordingSession:
    def __init__(self, posargs: list[str]) -> None:
        self.posargs = posargs
        self.env: dict[str, str] = {}
        self.runs: list[tuple[str, ...]] = []

    def run(self, *args: str, **kwargs: object) -> None:
        self.runs.append(args)

    def error(self, message: str) -> None:
        raise AssertionError(message)


# Module scope: nox warns when a noxfile registers the same session twice.
@pytest.fixture(scope="module")
def noxfile_module():
    spec = importlib.util.spec_from_file_location("noxfile_under_test", NOXFILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def noxfile(noxfile_module, monkeypatch):
    monkeypatch.setattr(noxfile_module, "install_locked", lambda *args, **kwargs: None)
    monkeypatch.chdir(NOXFILE.parent)
    return noxfile_module


class TestNotebookSessions:
    @pytest.mark.parametrize("session", ["examples", "examples-smoke"])
    def test_missing_smoke_notebook_fails_the_session(self, tmp_path, session):
        shutil.copyfile(NOXFILE, tmp_path / "noxfile.py")
        examples = tmp_path / "docs/source/examples"
        examples.mkdir(parents=True)
        (examples / "unrelated.ipynb").touch()

        result = subprocess.run(
            [sys.executable, "-m", "nox", "--no-venv", "-s", session],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "Missing smoke notebooks:" in result.stderr

    @pytest.mark.parametrize("session", ["run_examples", "run_examples_smoke"])
    def test_posargs_replace_the_default_notebooks(self, noxfile, session):
        notebook = str(noxfile.SMOKE_NOTEBOOKS[0])
        recorder = RecordingSession([notebook])

        getattr(noxfile, session)(recorder)

        assert recorder.runs == [("pytest", "--nbmake", notebook)]

    def test_examples_defaults_to_every_notebook(self, noxfile):
        recorder = RecordingSession([])

        noxfile.run_examples(recorder)

        assert recorder.runs == [("pytest", "--nbmake", str(noxfile.EXAMPLES_DIR))]

    def test_examples_smoke_defaults_to_smoke_notebooks(self, noxfile):
        recorder = RecordingSession([])

        noxfile.run_examples_smoke(recorder)

        expected = tuple(str(path) for path in noxfile.SMOKE_NOTEBOOKS)
        assert recorder.runs == [("pytest", "--nbmake", *expected)]
