#
# Tests for Ai (2020) Enertech parameter set loads
#
import pybamm
import unittest


class TestAi2020(unittest.TestCase):
    def test_load_params(self):
        negative_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/negative_electrodes/graphite_Ai2020/"
                "parameters.csv"
            )
        )
        self.assertEqual(negative_electrode["Negative electrode porosity"], "0.33")

        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/positive_electrodes/lico2_Ai2020/"
                "parameters.csv"
            )
        )
        self.assertEqual(positive_electrode["Positive electrode porosity"], "0.32")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/electrolytes/lipf6_Enertech_Ai2020/"
                + "parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.38")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/Enertech_Ai2020/parameters.csv"
            )
        )
        self.assertAlmostEqual(cell["Negative current collector thickness [m]"], 10e-6)

    def test_standard_lithium_parameters(self):
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        options = {"particle mechanics": "swelling and cracking"}
        model = pybamm.lithium_ion.DFN(options)
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
