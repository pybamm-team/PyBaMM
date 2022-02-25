#
# Tests for LG M50 parameter set loads
#
import pybamm
import unittest


class TestXu(unittest.TestCase):
    def test_load_params(self):
        negative_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/negative_electrodes/li_metal_Xu2019/"
                "parameters.csv"
            )
        )
        self.assertEqual(
            negative_electrode["Typical plated lithium concentration [mol.m-3]"],
            "76900",
        )

        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/positive_electrodes/NMC532_Xu2019/"
                "parameters.csv"
            )
        )
        self.assertEqual(positive_electrode["Positive electrode porosity"], "0.331")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/electrolytes/lipf6_Valoen2005/"
                + "parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.38")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/li_metal_Xu2019/parameters.csv"
            )
        )
        self.assertAlmostEqual(cell["Negative electrode thickness [m]"], 700e-6)

    def test_standard_lithium_parameters(self):

        parameter_values = pybamm.ParameterValues("Xu2019")

        model = pybamm.lithium_ion.BasicDFNHalfCell({"working electrode": "positive"})
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
