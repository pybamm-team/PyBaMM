#
# Tests for LG M50 parameter set loads
#
import pybamm
import unittest


class TestChen(unittest.TestCase):
    def test_load_params(self):
        anode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/anodes/graphite_Chen2020/parameters.csv"
            )
        )
        self.assertEqual(anode["Negative electrode porosity"], "0.25")

        cathode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/cathodes/nmc_Chen2020/parameters.csv"
            )
        )
        self.assertEqual(cathode["Positive electrode porosity"], "0.335")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/electrolytes/lipf6_Nyman2008/"
                + "parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.2594")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/cells/LGM50_Chen2020/parameters.csv"
            )
        )
        self.assertAlmostEqual(cell["Negative current collector thickness [m]"], 12e-6)

    def test_standard_lithium_parameters(self):

        chemistry = pybamm.parameter_sets.Chen2020
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
