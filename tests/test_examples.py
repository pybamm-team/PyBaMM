import os
import runpy

import pytest


class TestExamples:
    """
    A class to test the example scripts.
    """

    def list_of_files():
        file_list = []
        base_dir = os.path.join(
                os.path.dirname(__file__), "..", "examples", "scripts"
        )
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_list.append(os.path.join(root, file))
        return file_list



    @pytest.mark.parametrize("files", list_of_files())
    def test_example_scripts(self, files):
        runpy.run_path(files)

