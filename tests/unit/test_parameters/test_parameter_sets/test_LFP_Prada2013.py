#
# Tests for LFP Prada 2013 parameter set
#
import pybamm
import unittest


class TestLFPPrada2013(unittest.TestCase):
    def test_load_params(self):
        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/positive_electrodes/LFP_Prada2013/"
                "parameters.csv"
            )
        )
        self.assertEqual(
            positive_electrode["Positive electrode porosity"], "0.12728395"
        )

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/A123_Lain2019/parameters.csv"
            )
        )
        self.assertAlmostEqual(cell["Negative current collector thickness [m]"], 1e-5)

    def test_standard_lithium_parameters(self):

        chemistry = pybamm.parameter_sets.Prada2013
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)

        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sim.set_parameters()
        sim.build()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
