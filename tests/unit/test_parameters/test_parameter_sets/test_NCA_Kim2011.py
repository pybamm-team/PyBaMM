#
# Tests for NCA parameter set loads
#
import pybamm
import unittest


class TestKim(unittest.TestCase):
    def test_load_params(self):
        anode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/anodes/graphite_Kim2011/parameters.csv"
            )
        )
        self.assertEqual(anode["Negative electrode porosity"], "0.4")

        cathode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/cathodes/nca_Kim2011/parameters.csv"
            )
        )
        self.assertEqual(cathode["Positive electrode porosity"], "0.4")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/electrolytes/lipf6_Kim2011/parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.4")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/cells/Kim2011/parameters.csv"
            )
        )
        self.assertAlmostEqual(
            cell["Negative current collector thickness [m]"], 10 ** (-5)
        )

    def test_standard_lithium_parameters(self):

        chemistry = pybamm.parameter_sets.NCA_Kim2011
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
