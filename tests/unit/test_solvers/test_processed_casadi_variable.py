#
# Tests for the Processed Variable class
#
import pybamm
import casadi

import numpy as np
import unittest
import tests


class TestProcessedCasadiVariable(unittest.TestCase):
    def test_processed_variable_0D(self):
        # without inputs
        y = pybamm.StateVector(slice(0, 1))
        var = 2 * y
        var.mesh = None

        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        solution = pybamm.Solution(t_sol, y_sol)
        processed_var = pybamm.ProcessedCasadiVariable(var, solution)
        np.testing.assert_array_equal(processed_var.value(), 2 * y_sol)

        # No sensitivity as variable is not symbolic
        with self.assertRaisesRegex(ValueError, "Variable is not symbolic"):
            processed_var.sensitivity()

    def test_processed_variable_0D_with_inputs(self):
        # with symbolic inputs
        y = pybamm.StateVector(slice(0, 1))
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        var = p * y + q
        var.mesh = None

        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        solution = pybamm.Solution(t_sol, y_sol)
        solution.inputs = {"p": casadi.MX.sym("p"), "q": casadi.MX.sym("q")}
        processed_var = pybamm.ProcessedCasadiVariable(var, solution)
        np.testing.assert_array_equal(
            processed_var.value({"p": 3, "q": 4}).full(), 3 * y_sol + 4
        )
        np.testing.assert_array_equal(processed_var.value([3, 4]).full(), 3 * y_sol + 4)
        np.testing.assert_array_equal(
            processed_var.sensitivity({"p": 3, "q": 4}).full(),
            np.c_[y_sol.T, np.ones_like(y_sol).T],
        )

        # via value_and_sensitivity
        val, sens = processed_var.value_and_sensitivity([3, 4])
        np.testing.assert_array_equal(val.full(), 3 * y_sol + 4)
        np.testing.assert_array_equal(
            sens.full(), np.c_[y_sol.T, np.ones_like(y_sol).T]
        )

        # Test bad keys
        with self.assertRaisesRegex(ValueError, "Inconsistent input keys"):
            processed_var.value({"not p": 3})

    def test_processed_variable_0D_some_inputs(self):
        # with some symbolic inputs and some non-symbolic inputs
        y = pybamm.StateVector(slice(0, 1))
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        var = p * y - q
        var.mesh = None

        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        solution = pybamm.Solution(t_sol, y_sol)
        solution.inputs = {"p": casadi.MX.sym("p"), "q": 2}
        processed_var = pybamm.ProcessedCasadiVariable(var, solution)
        np.testing.assert_array_equal(
            processed_var.value({"p": 3}).full(), 3 * y_sol - 2
        )
        np.testing.assert_array_equal(
            processed_var.sensitivity({"p": 3}).full(), y_sol.T
        )

    def test_processed_variable_1D(self):
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = var + x

        # On nodes
        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        eqn_sol = disc.process_symbol(eqn)

        # With scalar t_sol
        t_sol = [0]
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * 5
        sol = pybamm.Solution(t_sol, y_sol)
        processed_eqn = pybamm.ProcessedCasadiVariable(eqn_sol, sol)
        np.testing.assert_array_equal(
            processed_eqn.value(), y_sol + x_sol[:, np.newaxis]
        )

        # With vector t_sol
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)
        sol = pybamm.Solution(t_sol, y_sol)
        processed_eqn = pybamm.ProcessedCasadiVariable(eqn_sol, sol)
        np.testing.assert_array_equal(
            processed_eqn.value(), y_sol + x_sol[:, np.newaxis]
        )

    def test_processed_variable_1D_with_scalar_inputs(self):
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        eqn = var * p + 2 * q

        # On nodes
        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        eqn_sol = disc.process_symbol(eqn)

        # Scalar t
        t_sol = [0]
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * 5

        sol = pybamm.Solution(t_sol, y_sol)
        sol.inputs = {"p": casadi.MX.sym("p"), "q": casadi.MX.sym("q")}
        processed_eqn = pybamm.ProcessedCasadiVariable(eqn_sol, sol)

        # Test values
        np.testing.assert_array_equal(processed_eqn.value([2, 3]), 2 * y_sol + 6)
        np.testing.assert_array_equal(
            processed_eqn.value({"p": 27, "q": -42}), 27 * y_sol - 84,
        )

        # Test sensitivities
        np.testing.assert_array_equal(
            processed_eqn.sensitivity({"p": 27, "q": -84}),
            np.c_[y_sol, 2 * np.ones_like(y_sol)],
        )

        ################################################################################
        # Vector t
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)

        sol = pybamm.Solution(t_sol, y_sol)
        sol.inputs = {"p": casadi.MX.sym("p"), "q": casadi.MX.sym("q")}
        processed_eqn = pybamm.ProcessedCasadiVariable(eqn_sol, sol)

        # Test values
        np.testing.assert_array_equal(processed_eqn.value([2, 3]), 2 * y_sol + 6)
        np.testing.assert_array_equal(
            processed_eqn.value({"p": 27, "q": -42}), 27 * y_sol - 84,
        )

        # Test sensitivities
        np.testing.assert_array_equal(
            processed_eqn.sensitivity({"p": 27, "q": -42}),
            np.c_[y_sol.T.flatten(), 2 * np.ones_like(y_sol.T.flatten())],
        )

    def test_processed_variable_1D_with_vector_inputs(self):
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        eqn = (var * p) ** 2 + 2 * q

        # On nodes
        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        n = x_sol.size
        eqn_sol = disc.process_symbol(eqn)

        # Scalar t
        t_sol = [0]
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * 5
        sol = pybamm.Solution(t_sol, y_sol)
        sol.inputs = {"p": casadi.MX.sym("p", n), "q": casadi.MX.sym("q")}
        processed_eqn = pybamm.ProcessedCasadiVariable(eqn_sol, sol)

        # Test values - constant p
        np.testing.assert_array_equal(
            processed_eqn.value(np.r_[2 * np.ones(n), [3]]), (2 * y_sol) ** 2 + 6
        )
        np.testing.assert_array_equal(
            processed_eqn.value({"p": 27 * np.ones(n), "q": -42}),
            (27 * y_sol) ** 2 - 84,
        )
        # Test values - varying p
        p = np.linspace(0, 1, n)
        np.testing.assert_array_equal(
            processed_eqn.value(np.r_[p, [3]]), (p[:, np.newaxis] * y_sol) ** 2 + 6,
        )

        # Test sensitivities - constant p
        np.testing.assert_array_equal(
            processed_eqn.sensitivity({"p": 2 * np.ones(n), "q": -84}),
            np.c_[100 * np.eye(y_sol.size), 2 * np.ones(n)],
        )
        # Test sensitivities - varying p
        # d/dy((py)**2) = (2*p*y) * y
        np.testing.assert_array_equal(
            processed_eqn.sensitivity({"p": p, "q": -84}),
            np.c_[
                np.diag((2 * p[:, np.newaxis] * y_sol ** 2).flatten()), 2 * np.ones(n)
            ],
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
