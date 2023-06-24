#
# Tests the utility functions.
#
from tests import TestCase
import numpy as np
import os
import sys
import pybamm
import tempfile
import unittest
from unittest.mock import patch
from io import StringIO


class TestUtil(TestCase):
    """
    Test the functionality in util.py
    """

    def test_rmse(self):
        self.assertEqual(pybamm.rmse(np.ones(5), np.zeros(5)), 1)
        self.assertEqual(pybamm.rmse(2 * np.ones(5), np.zeros(5)), 2)
        self.assertEqual(pybamm.rmse(2 * np.ones(5), np.ones(5)), 1)

        x = np.array([1, 2, 3, 4, 5])
        self.assertEqual(pybamm.rmse(x, x), 0)

        with self.assertRaisesRegex(ValueError, "same length"):
            pybamm.rmse(np.ones(5), np.zeros(3))

    def test_is_constant_and_can_evaluate(self):
        symbol = pybamm.PrimaryBroadcast(0, "negative electrode")
        self.assertEqual(False, pybamm.is_constant_and_can_evaluate(symbol))
        symbol = pybamm.StateVector(slice(0, 1))
        self.assertEqual(False, pybamm.is_constant_and_can_evaluate(symbol))
        symbol = pybamm.Scalar(0)
        self.assertEqual(True, pybamm.is_constant_and_can_evaluate(symbol))

    def test_fuzzy_dict(self):
        d = pybamm.FuzzyDict(
            {
                "test": 1,
                "test2": 2,
                "SEI current": 3,
                "Lithium plating current": 4,
                "A dimensional variable [m]": 5,
            }
        )
        self.assertEqual(d["test"], 1)
        with self.assertRaisesRegex(KeyError, "'test3' not found. Best matches are "):
            d.__getitem__("test3")

        with self.assertRaisesRegex(KeyError, "stoichiometry"):
            d.__getitem__("Negative electrode SOC")

        with self.assertRaisesRegex(KeyError, "dimensional version"):
            d.__getitem__("A dimensional variable")

        with self.assertRaisesRegex(KeyError, "open circuit voltage"):
            d.__getitem__("Measured open circuit voltage [V]")

        with self.assertRaisesRegex(KeyError, "Lower voltage"):
            d.__getitem__("Open-circuit voltage at 0% SOC [V]")

        with self.assertRaisesRegex(KeyError, "Upper voltage"):
            d.__getitem__("Open-circuit voltage at 100% SOC [V]")

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


class TestSearch(TestCase):
    def test_url_gets_to_stdout(self):
        model = pybamm.BaseModel()
        model.variables = {"Electrolyte concentration": 1, "Electrode potential": 0}

        param = pybamm.ParameterValues({"test": 10, "b": 2})

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
            param.search("test")
            self.assertEqual(fake_out.getvalue(), "test\t10\n")


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
