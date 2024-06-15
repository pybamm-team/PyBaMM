import os
import runpy

import pytest
from pathlib import Path


ROOT_DIR = Path(os.path.join(os.path.dirname(__file__), "..")
class TestExamples:
    """
    A class to test the example scripts.
    """

    def list_of_files():
        base_dir = Path.joinpath(ROOT_DIR, "examples", "scripts")
        file_list = base_dir.rglob("*py")
        return file_list

    @pytest.mark.parametrize("files", list_of_files())
    @pytest.mark.examples
    def test_example_scripts(self, files):
        runpy.run_path(files)
