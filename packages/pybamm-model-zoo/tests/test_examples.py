"""Run every model's example scripts, mirroring the core package's test_scripts."""

import runpy

import pytest

import pybamm_model_zoo as zoo
from pybamm_model_zoo.testing import contract


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
        runpy.run_path(str(script))
