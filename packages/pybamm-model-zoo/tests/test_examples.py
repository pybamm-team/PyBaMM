"""Run every model's example scripts, mirroring the core package's test_scripts."""

import runpy

import pytest

import pybamm_model_zoo as zoo
from pybamm_model_zoo.testing import contract


def run_example(script):
    """Execute an example script the way a user would, as ``__main__``."""
    runpy.run_path(str(script), run_name="__main__")


def example_scripts():
    return [
        pytest.param(
            entry,
            script,
            id=f"{entry.slug}/{script.name}",
            marks=pytest.mark.zoo_model(entry.slug),
        )
        for entry in zoo.all_entries()
        for script in sorted((entry.path / "examples").glob("**/*.py"))
    ]


class TestExamples:
    @pytest.mark.zoo_examples
    @pytest.mark.parametrize(("entry", "script"), example_scripts())
    def test_example_script(self, entry, script):
        if missing := contract.missing_dependencies(entry):
            pytest.skip(f"{entry.slug}: missing {missing}")
        run_example(script)

    def test_a_main_guarded_example_is_executed(self, tmp_path):
        script = tmp_path / "guarded.py"
        ran = tmp_path / "ran.txt"
        script.write_text(
            "from pathlib import Path\n\n\n"
            "def main():\n"
            f"    Path({str(ran)!r}).write_text('ran')\n\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n",
            encoding="utf-8",
        )
        run_example(script)
        assert ran.read_text(encoding="utf-8") == "ran"
