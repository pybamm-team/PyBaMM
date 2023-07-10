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

    def test_set_msmr_variables(self):
        with self.assertRaisesRegex(pybamm.OptionError, "MSMR"):
            pybamm.lithium_ion.BaseModel().set_msmr_reaction_variables(None)

        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
        }
        model = pybamm.lithium_ion.SPM(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")
        model.set_msmr_reaction_variables(parameter_values)
        xn_2 = model.variables["x2_n"]
        # For SPM, xn_2 will be a broadcast of the reaction formula, whose child should
        # be the parameter "Xj_n_2"
        self.assertIsInstance(xn_2.children[0].children[0], pybamm.Parameter)
        self.assertEqual(xn_2.children[0].children[0].name, "Xj_n_2")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
