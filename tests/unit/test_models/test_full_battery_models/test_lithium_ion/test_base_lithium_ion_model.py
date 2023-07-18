#
# Tests for the base lead acid model class
#
from tests import TestCase
import pybamm
import unittest
import os


class TestBaseLithiumIonModel(TestCase):
    def test_incompatible_options(self):
        with self.assertRaisesRegex(pybamm.OptionError, "convection not implemented"):
            pybamm.lithium_ion.BaseModel({"convection": "uniform transverse"})

    def test_default_parameters(self):
        # check parameters are read in ok
        model = pybamm.lithium_ion.BaseModel()
        self.assertEqual(
            model.default_parameter_values["Reference temperature [K]"], 298.15
        )

        # change path and try again

        cwd = os.getcwd()
        os.chdir("..")
        model = pybamm.lithium_ion.BaseModel()
        self.assertEqual(
            model.default_parameter_values["Reference temperature [K]"], 298.15
        )
        os.chdir(cwd)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
