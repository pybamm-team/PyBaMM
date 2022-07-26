#
# Tests for O'Kane (2022) parameter set
#
import pybamm
import unittest
import os


class TestOKane2022(unittest.TestCase):
    def test_load_params(self):
        Li_plating = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/lithium_platings/okane2022_Li_plating/"
                "parameters.csv"
            )
        )
        self.assertEqual(
            Li_plating["Lithium metal partial molar volume [m3.mol-1]"], "1.30E-05"
        )

    def test_functions(self):
        root = pybamm.root_dir()
        param = pybamm.ParameterValues("OKane2022")
        sto = pybamm.Scalar(0.9)
        T = pybamm.Scalar(298.15)

        # Lithium plating
        p = "pybamm/input/parameters/lithium_ion/lithium_platings/okane2022_Li_plating/"
        k_path = os.path.join(root, p)

        fun_test = {
            "plating_exchange_current_density_OKane2020.py": ([1e3, 1e4, T], 9.6485e-2),
            "stripping_exchange_current_density_OKane2020.py": (
                [1e3, 1e4, T],
                9.6485e-1,
            ),
            "SEI_limited_dead_lithium_OKane2022.py": ([1e-8], 5e-7),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

        # Negative electrode
        p = (
            "pybamm/input/parameters/lithium_ion/negative_electrodes/"
            "graphite_OKane2022/"
        )
        k_path = os.path.join(root, p)

        fun_test = {
            "graphite_LGM50_diffusivity_Chen2020.py": ([sto, T], 3.3e-14),
            "graphite_LGM50_electrolyte_exchange_current_density_Chen2020.py": (
                [1000, 16566.5, 33133, T],
                0.33947,
            ),
            "graphite_LGM50_ocp_Chen2020.py": ([sto], 0.0861),
            "graphite_cracking_rate_Ai2020.py": ([T], 3.9e-20),
            "graphite_volume_change_Ai2020.py": ([sto, 33133], 0.0897),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

        # Positive electrode
        p = "pybamm/input/parameters/lithium_ion/positive_electrodes/nmc_OKane2022/"
        k_path = os.path.join(root, p)

        fun_test = {
            "nmc_LGM50_diffusivity_Chen2020.py": ([sto, T], 4e-15),
            "nmc_LGM50_electrolyte_exchange_current_density_Chen2020.py": (
                [1000, 31552, 63104, T],
                3.4123,
            ),
            "nmc_LGM50_ocp_Chen2020.py": ([sto], 3.5682),
            "cracking_rate_Ai2020.py": ([T], 3.9e-20),
            "volume_change_Ai2020.py": ([sto, 63104], 0.70992),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
