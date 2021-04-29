#
# Tests for  Yang2017 parameter set loads
#
import pybamm
import unittest


class TestYang2017(unittest.TestCase):
    def test_load_params(self):
        negative_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/negative_electrodes/graphite_Yang2017/"
                "parameters.csv"
            )
        )
        self.assertEqual(negative_electrode["Negative electrode porosity"], "0.32")

        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/positive_electrodes/nmc_Yang2017/"
                "parameters.csv"
            )
        )
        self.assertEqual(positive_electrode["Positive electrode porosity"], "0.33")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/electrolytes/lipf6_Ecker2015/"
                + "parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.26")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/Yang2017/parameters.csv"
            )
        )
        self.assertAlmostEqual(cell["Negative current collector thickness [m]"], 25e-6)

    def test_standard_lithium_parameters(self):

        chemistry = pybamm.parameter_sets.Yang2017
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)

        model = pybamm.lithium_ion.Yang2017()
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
