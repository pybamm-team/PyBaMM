#
# Tests for Ecker parameter set
#
import pybamm
import unittest


class TestEcker(unittest.TestCase):
    def test_load_params(self):
        anode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/anodes/graphite_Ecker2015/parameters.csv"
            )
        )
        self.assertEqual(anode["Negative electrode porosity"], "0.329")

        path = "input/parameters/lithium-ion/cathodes/LiNiCoO2_Ecker2015/parameters.csv"
        cathode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(path)
        )
        self.assertEqual(cathode["Positive electrode conductivity [S.m-1]"], "68.1")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/electrolytes/lipf6_Ecker2015/"
                + "parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.26")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/cells/kokam_Ecker2015/parameters.csv"
            )
        )
        self.assertAlmostEqual(cell["Negative current collector thickness [m]"], 14e-6)

    def test_standard_lithium_parameters(self):

        chemistry = pybamm.parameter_sets.Ecker2015
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
