#
# Tests for NCA parameter set loads
#
import pybamm
import unittest


class TestKim(unittest.TestCase):
    def test_load_params(self):
        negative_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/negative_electrodes/graphite_Kim2011/"
                "parameters.csv"
            )
        )
        self.assertEqual(negative_electrode["Negative electrode porosity"], "0.4")

        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/positive_electrodes/nca_Kim2011/"
                "parameters.csv"
            )
        )
        self.assertEqual(positive_electrode["Positive electrode porosity"], "0.4")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/electrolytes/lipf6_Kim2011/parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.4")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/Kim2011/parameters.csv"
            )
        )
        self.assertAlmostEqual(
            cell["Negative current collector thickness [m]"], 10 ** (-5)
        )

    def test_standard_lithium_parameters(self):

        parameter_values = pybamm.ParameterValues("NCA_Kim2011")

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
