#
# Tests the utility functions.
#
import numpy as np
import os
import pybamm
import shutil
import tempfile
import unittest
import importlib
import subprocess
from unittest.mock import patch
from io import StringIO

# Insert .../x/y/z/PyBaMM in sys.path when running this file individually
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


class TestUtil(unittest.TestCase):
    """
    Test the functionality in util.py
    """

    def test_load_function(self):
        # Test function load with absolute path
        abs_test_path = os.path.join(
            pybamm.root_dir(),
            "pybamm",
            "input",
            "parameters",
            "lithium_ion",
            "negative_electrodes",
            "graphite_Chen2020",
            "graphite_LGM50_electrolyte_exchange_current_density_Chen2020.py",
        )
        func = pybamm.load_function(abs_test_path)
        self.assertEqual(
            func,
            pybamm.input.parameters.lithium_ion.negative_electrodes.graphite_Chen2020.graphite_LGM50_electrolyte_exchange_current_density_Chen2020.graphite_LGM50_electrolyte_exchange_current_density_Chen2020,  # noqa
        )

        # Test function load with relative path
        rel_test_path = os.path.join(
            "pybamm",
            "input",
            "parameters",
            "lithium_ion",
            "negative_electrodes",
            "graphite_Chen2020",
            "graphite_LGM50_electrolyte_exchange_current_density_Chen2020.py",
        )
        func = pybamm.load_function(rel_test_path)
        self.assertEqual(
            func,
            pybamm.input.parameters.lithium_ion.negative_electrodes.graphite_Chen2020.graphite_LGM50_electrolyte_exchange_current_density_Chen2020.graphite_LGM50_electrolyte_exchange_current_density_Chen2020,  # noqa
        )

        # Test function load for parameters in a directory having "pybamm" in its name
        # create a new lithium_ion folder in the root PyBaMM directory
        subprocess.run(["pybamm_edit_parameter", "lithium_ion"])

        # path for a function in the created directory ->
        # x/y/z/PyBaMM/lithium_ion/negative_electrode/ ....
        test_path = os.path.join(
            os.getcwd(),
            "lithium_ion",
            "negative_electrodes",
            "graphite_Chen2020",
            "graphite_LGM50_electrolyte_exchange_current_density_Chen2020.py",
        )

        # load the function
        func = pybamm.load_function(test_path)

        # cannot directly do - lithium_ion.negative_electrodes.graphite_Chen2020 as
        # lithium_ion is not a python module
        module_object = importlib.import_module(
            "lithium_ion.negative_electrodes.graphite_Chen2020.graphite_LGM50_electrolyte_exchange_current_density_Chen2020"  # noqa
        )
        self.assertEqual(
            func,
            getattr(
                module_object,
                "graphite_LGM50_electrolyte_exchange_current_density_Chen2020",
            ),
        )

        shutil.rmtree("lithium_ion")

    def test_rmse(self):
        self.assertEqual(pybamm.rmse(np.ones(5), np.zeros(5)), 1)
        self.assertEqual(pybamm.rmse(2 * np.ones(5), np.zeros(5)), 2)
        self.assertEqual(pybamm.rmse(2 * np.ones(5), np.ones(5)), 1)

        x = np.array([1, 2, 3, 4, 5])
        self.assertEqual(pybamm.rmse(x, x), 0)

        with self.assertRaisesRegex(ValueError, "same length"):
            pybamm.rmse(np.ones(5), np.zeros(3))

    def test_infinite_nested_dict(self):
        d = pybamm.get_infinite_nested_dict()
        d[1][2][3] = "x"
        self.assertEqual(d[1][2][3], "x")
        d[4][5] = "y"
        self.assertEqual(d[4][5], "y")

    def test_fuzzy_dict(self):
        d = pybamm.FuzzyDict(
            {
                "test": 1,
                "test2": 2,
                "SEI current": 3,
                "Lithium plating current": 4,
            }
        )
        self.assertEqual(d["test"], 1)
        with self.assertRaisesRegex(KeyError, "'test3' not found. Best matches are "):
            d.__getitem__("test3")
        with self.assertRaisesRegex(
            KeyError, "'negative electrode SEI current' not found. All SEI parameters"
        ):
            d.__getitem__("negative electrode SEI current")
        with self.assertRaisesRegex(
            KeyError,
            "'negative electrode lithium plating current' not found. "
            "All lithium plating parameters",
        ):
            d.__getitem__("negative electrode lithium plating current")

    def test_get_parameters_filepath(self):
        tempfile_obj = tempfile.NamedTemporaryFile("w", dir=".")
        self.assertTrue(
            pybamm.get_parameters_filepath(tempfile_obj.name) == tempfile_obj.name
        )
        tempfile_obj.close()

        package_dir = os.path.join(pybamm.root_dir(), "pybamm")
        tempfile_obj = tempfile.NamedTemporaryFile("w", dir=package_dir)
        path = os.path.join(package_dir, tempfile_obj.name)
        self.assertTrue(pybamm.get_parameters_filepath(tempfile_obj.name) == path)
        tempfile_obj.close()

    def test_is_jax_compatible(self):
        if pybamm.have_jax():
            compatible = pybamm.is_jax_compatible()
            self.assertTrue(compatible)

    def test_git_commit_info(self):
        git_commit_info = pybamm.get_git_commit_info()
        self.assertIsInstance(git_commit_info, str)
        self.assertEqual(git_commit_info[:2], "v2")


class TestSearch(unittest.TestCase):
    def test_url_gets_to_stdout(self):
        model = pybamm.BaseModel()
        model.variables = {"Electrolyte concentration": 1, "Electrode potential": 0}

        param = pybamm.ParameterValues({"a": 10, "b": 2})

        # Test variables search (default returns key)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("Electrode")
            self.assertEqual(fake_out.getvalue(), "Electrode potential\n")

        # Test bad var search (returns best matches)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("Electrolyte cot")
            out = (
                "No results for search using 'Electrolyte cot'. "
                "Best matches are ['Electrolyte concentration', "
                "'Electrode potential']\n"
            )
            self.assertEqual(fake_out.getvalue(), out)

        # Test param search (default returns key, value)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            param.search("a")
            self.assertEqual(fake_out.getvalue(), "a\t10\n")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
