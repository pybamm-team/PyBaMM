import runpy

import pytest
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent


class TestExamples:
    """
    A class to test the example scripts.
    """

    def list_of_files():
        file_list = (ROOT_DIR / "examples" / "scripts").rglob("*.py")
        return [pytest.param(file, id=file.name) for file in file_list]

    @pytest.mark.parametrize("files", list_of_files())
    @pytest.mark.scripts
    def test_example_scripts(self, files):
        runpy.run_path(files)
