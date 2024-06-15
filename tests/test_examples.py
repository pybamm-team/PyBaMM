import os
import runpy
from pathlib import Path
import pytest

ROOT_DIR = Path(os.path.join(
    os.path.dirname(__file__), ".."))


class TestExamples:
    """
    A class to test the example scripts.
    """

    def list_of_files():
        base_dir = ROOT_DIR.joinpath("examples", "scripts")
        # Recursively find all python files inside examples/scripts
        file_list = list(base_dir.rglob('*.py'))
        return file_list

    @pytest.mark.parametrize("files", list_of_files())
    @pytest.mark.examples
    def test_example_scripts(self, files):
        runpy.run_path(files)
