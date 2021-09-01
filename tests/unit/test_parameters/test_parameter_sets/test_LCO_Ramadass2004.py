#
# Tests for Ai (2020) Enertech parameter set loads
#
import pybamm
import unittest
import os


class TestRamadass2004(unittest.TestCase):
    def test_load_params(self):
        negative_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/negative_electrodes/"
                "graphite_Ramadass2004/parameters.csv"
            )
        )
        self.assertEqual(negative_electrode["Negative electrode porosity"], "0.485")

        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/positive_electrodes/lico2_Ramadass2004/"
                "parameters.csv"
            )
        )
        self.assertEqual(positive_electrode["Positive electrode porosity"], "0.385")

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/electrolytes/lipf6_Ramadass2004/"
                + "parameters.csv"
            )
        )
        self.assertEqual(electrolyte["Cation transference number"], "0.363")

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/sony_Ramadass2004/parameters.csv"
            )
        )
        self.assertAlmostEqual(
            cell["Negative current collector thickness [m]"],
            1.7E-05
        )

    def test_functions(self):
        root = pybamm.root_dir()
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ramadass2004)
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)

        # Positive electrode
        p = (
            "pybamm/input/parameters/lithium_ion/positive_electrodes/"
            "lico2_Ramadass2004/"
        )
        k_path = os.path.join(root, p)

        fun_test = {
            "lico2_diffusivity_Ramadass2004.py": ([sto, T], 1e-14),
            "lico2_electrolyte_exchange_current_density_Ramadass2004.py": (
                [1e3, 1e4, T],
                1.4517,
            ),
            "lico2_entropic_change_Moura2016.py": ([sto], -3.4664e-5),
            "lico2_ocp_Ramadass2004.py": ([sto], 4.1249),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

        # Negative electrode
        p = (
            "pybamm/input/parameters/lithium_ion/negative_electrodes/"
            "graphite_Ramadass2004/"
        )
        k_path = os.path.join(root, p)

        fun_test = {
            "graphite_mcmb2528_diffusivity_Dualfoil1998.py": ([sto, T], 3.9e-14),
            "graphite_electrolyte_exchange_current_density_Ramadass2004.py": (
                [1e3, 1e4, T],
                2.2007,
            ),
            "graphite_entropic_change_Moura2016.py": ([sto], -1.5079e-5),
            "graphite_ocp_Ramadass2004.py": ([sto], 0.1215),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

    def test_standard_lithium_parameters(self):
        chemistry = pybamm.parameter_sets.Ramadass2004
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
