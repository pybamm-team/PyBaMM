#
# Tests for Ai (2020) Enertech parameter set loads
#
import pybamm
import unittest
import os


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

    def test_functions(self):
        root = pybamm.root_dir()
        param = pybamm.ParameterValues("Ai2020")
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)

        # Positive electrode
        p = "pybamm/input/parameters/lithium_ion/positive_electrodes/lico2_Ai2020/"
        k_path = os.path.join(root, p)

        fun_test = {
            "lico2_cracking_rate_Ai2020": ([T], 3.9e-20),
            "lico2_diffusivity_Dualfoil1998": ([sto, T], 5.387e-15),
            "lico2_electrolyte_exchange_current_density_Dualfoil1998": (
                [1e3, 1e4, T],
                0.6098,
            ),
            "lico2_entropic_change_Ai2020_function": ([sto], -2.1373e-4),
            "lico2_ocp_Ai2020_function.py": ([sto], 4.1638),
            "lico2_volume_change_Ai2020": ([sto], -1.8179e-2),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

        # Negative electrode
        p = "pybamm/input/parameters/lithium_ion/negative_electrodes/graphite_Ai2020/"
        k_path = os.path.join(root, p)

        fun_test = {
            "graphite_cracking_rate_Ai2020.py": ([T], 3.9e-20),
            "graphite_diffusivity_Dualfoil1998.py": ([sto, T], 3.9e-14),
            "graphite_electrolyte_exchange_current_density_Dualfoil1998.py": (
                [1e3, 1e4, T],
                0.4172,
            ),
            "graphite_entropy_Enertech_Ai2020_function.py": ([sto], -1.1033e-4),
            "graphite_ocp_Enertech_Ai2020_function.py": ([sto], 0.1395),
            "graphite_volume_change_Ai2020.py": ([sto], 5.1921e-2),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

    def test_standard_lithium_parameters(self):
        parameter_values = pybamm.ParameterValues("Ai2020")
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
