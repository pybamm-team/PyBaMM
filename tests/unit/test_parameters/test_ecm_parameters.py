#
# Tests for the equivalent circuit parameters
#
from tests import TestCase
import pybamm
import unittest


values = {
    "Initial SoC": 0.5,
    "Initial temperature [K]": 25 + 273.15,
    "Cell capacity [A.h]": 100,
    "Nominal cell capacity [A.h]": 100,
    "Ambient temperature [K]": 25 + 273.15,
    "Current function [A]": 100,
    "Upper voltage cut-off [V]": 4.2,
    "Lower voltage cut-off [V]": 3.2,
    "Open-circuit voltage at 0% SOC [V]": 2.8,
    "Open-circuit voltage at 100% SOC [V]": 4.2,
    "Cell thermal mass [J/K]": 1000,
    "Cell-jig heat transfer coefficient [W/K]": 10,
    "Jig thermal mass [J/K]": 500,
    "Jig-air heat transfer coefficient [W/K]": 10,
    "R0 [Ohm]": 0.4e-3,
    "Element-1 initial overpotential [V]": 0,
    "R1 [Ohm]": 0.6e-3,
    "C1 [F]": 30 / 0.6e-3,
    "Entropic change [V/K]": 0,
    "RCR lookup limit [A]": 340,
    "Open-circuit voltage [V]": 3.4,
}

parameter_values = pybamm.ParameterValues(values)


class TestEcmParameters(TestCase):
    def test_init_parameters(self):
        param = pybamm.EcmParameters()

        simpled_mapped_parameters = [
            (param.cell_capacity, "Cell capacity [A.h]"),
            (param.current_with_time, "Current function [A]"),
            (param.voltage_high_cut, "Upper voltage cut-off [V]"),
            (param.voltage_low_cut, "Lower voltage cut-off [V]"),
            (param.cth_cell, "Cell thermal mass [J/K]"),
            (param.k_cell_jig, "Jig-air heat transfer coefficient [W/K]"),
            (param.cth_jig, "Jig thermal mass [J/K]"),
            (param.k_jig_air, "Jig-air heat transfer coefficient [W/K]"),
            (param.Q, "Cell capacity [A.h]"),
            (param.current_density_with_time, "Current function [A]"),
            (param.initial_soc, "Initial SoC"),
        ]

        for symbol, key in simpled_mapped_parameters:
            value = parameter_values.evaluate(symbol)
            expected_value = values[key]
            self.assertEqual(value, expected_value)

        value = parameter_values.evaluate(param.initial_T_cell)
        self.assertEqual(value, values["Initial temperature [K]"] - 273.15)

        value = parameter_values.evaluate(param.initial_T_jig)
        self.assertEqual(value, values["Initial temperature [K]"] - 273.15)

        compatibility_parameters = [
            (param.n_electrodes_parallel, 1),
            (param.A_cc, 1),
            (param.n_cells, 1),
        ]

        for symbol, expected_value in compatibility_parameters:
            value = parameter_values.evaluate(symbol)
            self.assertEqual(value, expected_value)

    def test_function_parameters(self):
        param = pybamm.EcmParameters()

        sym = pybamm.Scalar(1)

        mapped_functions = [
            (param.ocv(sym), "Open-circuit voltage [V]"),
            (param.rcr_element("R0 [Ohm]", sym, sym, sym), "R0 [Ohm]"),
            (param.rcr_element("R1 [Ohm]", sym, sym, sym), "R1 [Ohm]"),
            (param.rcr_element("C1 [F]", sym, sym, sym), "C1 [F]"),
            (param.initial_rc_overpotential(1), "Element-1 initial overpotential [V]"),
            (param.dUdT(sym, sym), "Entropic change [V/K]"),
        ]

        for symbol, key in mapped_functions:
            value = parameter_values.evaluate(symbol)
            expected_value = values[key]
            self.assertEqual(value, expected_value)

        value = parameter_values.evaluate(param.T_amb(sym))
        self.assertEqual(value, values["Ambient temperature [K]"] - 273.15)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
