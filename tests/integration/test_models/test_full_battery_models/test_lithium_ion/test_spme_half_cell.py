#
# Tests for the half-cell lithium-ion SPMe model
#
import pybamm
import tests

import unittest


class TestSPMeHalfCell(unittest.TestCase):
    def test_basic_processing(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.SPMe(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Xu2019)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)


class TestSPMeHalfCellWithSEI(unittest.TestCase):
    def test_well_posed_constant(self):
        options = {"working electrode": "positive", "SEI": "constant"}
        model = pybamm.lithium_ion.SPMe(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Xu2019)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)

    def test_well_posed_reaction_limited(self):
        options = {"working electrode": "positive", "SEI": "reaction limited"}
        model = pybamm.lithium_ion.SPMe(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Xu2019)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)

    def test_well_posed_solvent_diffusion_limited(self):
        options = {"working electrode": "positive", "SEI": "solvent-diffusion limited"}
        model = pybamm.lithium_ion.SPMe(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Xu2019)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)

    def test_well_posed_electron_migration_limited(self):
        options = {"working electrode": "positive", "SEI": "electron-migration limited"}
        model = pybamm.lithium_ion.SPMe(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Xu2019)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)

    def test_well_posed_interstitial_diffusion_limited(self):
        options = {
            "working electrode": "positive",
            "SEI": "interstitial-diffusion limited",
        }
        model = pybamm.lithium_ion.SPMe(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Xu2019)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)

    def test_well_posed_ec_reaction_limited(self):
        options = {
            "working electrode": "positive",
            "SEI": "ec reaction limited",
        }
        model = pybamm.lithium_ion.SPMe(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Xu2019)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
