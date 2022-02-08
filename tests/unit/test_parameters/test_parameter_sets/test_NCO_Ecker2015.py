#
# Tests for Ecker parameter set
#
import pybamm
import unittest


class TestEcker(unittest.TestCase):
    def test_load_params(self):
        negative_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/negative_electrodes/graphite_Ecker2015/"
                "parameters.csv"
            )
        )
        self.assertEqual(negative_electrode["Negative electrode porosity"], "0.329")

        path = (
            "input/parameters/lithium_ion/positive_electrodes/LiNiCoO2_Ecker2015/"
            "parameters.csv"
        )
        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(path)
        )
        self.assertEqual(
            positive_electrode["Positive electrode conductivity [S.m-1]"], "68.1"
        )

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/electrolytes/lipf6_Ecker2015/"
                + "parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.26")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/kokam_Ecker2015/parameters.csv"
            )
        )
        self.assertAlmostEqual(cell["Negative current collector thickness [m]"], 14e-6)

    def test_standard_lithium_parameters(self):

        parameter_values = pybamm.ParameterValues("Ecker2015")

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
