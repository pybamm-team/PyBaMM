#
# Tests for LG M50 parameter set loads
#
import pybamm
import unittest
import os


class TestORegan2021(unittest.TestCase):
    def test_load_params(self):
        negative_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/negative_electrodes/graphite_ORegan2021/"
                "parameters.csv"
            )
        )
        self.assertEqual(
            negative_electrode["Negative electrode diffusivity [m2.s-1]"],
            "[function]graphite_LGM50_diffusivity_ORegan2021",
        )

        positive_electrode = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/positive_electrodes/nmc_ORegan2021/"
                "parameters.csv"
            )
        )
        self.assertEqual(
            positive_electrode["Positive electrode conductivity [S.m-1]"],
            "[function]nmc_LGM50_electronic_conductivity_ORegan2021",
        )

        electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/electrolytes/lipf6_EC_EMC_3_7_"
                "Landesfeind2019/parameters.csv"
            )
        )
        self.assertEqual(
            electrolyte["Typical electrolyte concentration [mol.m-3]"], "1000"
        )

        cell = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                "input/parameters/lithium_ion/cells/LGM50_ORegan2021/parameters.csv"
            )
        )
        self.assertEqual(
            cell["Negative current collector thermal conductivity [W.m-1.K-1]"],
            "[function]copper_thermal_conductivity_CRC",
        )

    def test_functions(self):
        root = pybamm.root_dir()
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.ORegan2021)
        T = pybamm.Scalar(298.15)

        # Positive electrode
        p = "pybamm/input/parameters/lithium_ion/positive_electrodes/nmc_ORegan2021/"
        k_path = os.path.join(root, p)

        fun_test = {
            "nmc_LGM50_entropic_change_ORegan2021.py": ([0.5], -9.7940e-07),
            "nmc_LGM50_heat_capacity_ORegan2021.py": ([298.15], 902.6502),
            "nmc_LGM50_diffusivity_ORegan2021.py": ([0.5, 298.15], 7.2627e-15),
            "nmc_LGM50_electrolyte_exchange_current_density_ORegan2021.py": (
                [1e3, 1e4, 298.15],
                2.1939,
            ),
            "nmc_LGM50_ocp_Chen2020.py": ([0.5], 3.9720),
            "nmc_LGM50_electronic_conductivity_ORegan2021.py": ([298.15], 0.8473),
            "nmc_LGM50_thermal_conductivity_ORegan2021.py": ([T], 0.8047),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

        # Negative electrode
        p = (
            "pybamm/input/parameters/lithium_ion/negative_electrodes/"
            "graphite_ORegan2021/"
        )
        k_path = os.path.join(root, p)

        fun_test = {
            "graphite_LGM50_entropic_change_ORegan2021.py": ([0.5], -2.6460e-07),
            "graphite_LGM50_heat_capacity_ORegan2021.py": ([298.15], 847.7155),
            "graphite_LGM50_diffusivity_ORegan2021.py": ([0.5, 298.15], 2.8655e-16),
            "graphite_LGM50_electrolyte_exchange_current_density_ORegan2021.py": (
                [1e3, 1e4, 298.15],
                1.0372,
            ),
            "graphite_LGM50_ocp_Chen2020.py": ([0.5], 0.1331),
            "graphite_LGM50_thermal_conductivity_ORegan2021.py": ([T], 3.7695),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

        # Cells
        p = "pybamm/input/parameters/lithium_ion/cells/LGM50_ORegan2021/"
        k_path = os.path.join(root, p)

        fun_test = {
            "aluminium_heat_capacity_CRC.py": ([T], 897.1585),
            "copper_heat_capacity_CRC.py": ([T], 388.5190),
            "copper_thermal_conductivity_CRC.py": ([T], 400.8491),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

        # Separator
        p = "pybamm/input/parameters/lithium_ion/separators/separator_ORegan2021/"
        k_path = os.path.join(root, p)

        fun_test = {
            "separator_LGM50_heat_capacity_ORegan2021.py": ([298.15], 1130.9656),
        }

        for name, value in fun_test.items():
            fun = pybamm.load_function(os.path.join(k_path, name))
            self.assertAlmostEqual(param.evaluate(fun(*value[0])), value[1], places=4)

    def test_standard_lithium_parameters(self):

        chemistry = pybamm.parameter_sets.ORegan2021
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
