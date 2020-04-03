#
# Tests for the Processed Variable class
#
import pybamm
import casadi

import numpy as np
import unittest


class TestProcessedCasadiVariable(unittest.TestCase):
    def test_processed_variable_0D(self):
        # without space
        y = pybamm.StateVector(slice(0, 1))
        var = 2 * y
        var.mesh = None

        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        solution = pybamm.CasadiSolution(t_sol, y_sol)
        processed_var = pybamm.ProcessedCasadiVariable(var, solution)
        np.testing.assert_array_equal(processed_var.value(), 2 * y_sol)

        # No sensitivity as variable is not symbolic
        with self.assertRaisesRegex(ValueError, "Variable is not symbolic"):
            processed_var.sensitivity()

    def test_processed_variable_0D_with_inputs(self):
        # without space
        y = pybamm.StateVector(slice(0, 1))
        p = pybamm.InputParameter("p")
        var = p * y
        var.mesh = None

        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        solution = pybamm.CasadiSolution(t_sol, y_sol)
        solution.inputs = {"p": casadi.MX.sym("p")}
        processed_var = pybamm.ProcessedCasadiVariable(var, solution)
        np.testing.assert_array_equal(processed_var.value({"p": 3}).full(), 3 * y_sol)
        np.testing.assert_array_equal(processed_var.value(3).full(), 3 * y_sol)
        np.testing.assert_array_equal(
            processed_var.sensitivity({"p": 3}).full(), y_sol.T
        )

        # via value_and_sensitivity
        val, sens = processed_var.value_and_sensitivity({"p": 3})
        np.testing.assert_array_equal(val.full(), 3 * y_sol)
        np.testing.assert_array_equal(sens.full(), y_sol.T)

        # Test bad keys
        with self.assertRaisesRegex(ValueError, "Inconsistent input keys"):
            processed_var.value({"not p": 3})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
