import runpy
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent


def list_of_files():
    file_list = (ROOT_DIR / "examples" / "scripts").rglob("*.py")
    return [pytest.param(file, id=file.name) for file in file_list]


class TestExamples:
    """
    A class to test the example scripts.
    """

    @pytest.mark.parametrize("files", list_of_files())
    @pytest.mark.scripts
    def test_example_scripts(self, files):
        runpy.run_path(str(files))
