#
# Tests for the Processed Variable class
#
import pybamm
import casadi

import numpy as np
import unittest
import tests


class TestProcessedSymbolicVariable(unittest.TestCase):
    def test_processed_variable_0D(self):
        # without inputs
        y = pybamm.StateVector(slice(0, 1))
        var = 2 * y
        var.mesh = None

        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        solution = pybamm.Solution(t_sol, y_sol)
        processed_var = pybamm.ProcessedSymbolicVariable(var, solution)
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
        processed_var = pybamm.ProcessedSymbolicVariable(var, solution)
        np.testing.assert_array_equal(
            processed_var.value({"p": 3, "q": 4}).full(), 3 * y_sol + 4
        )
        np.testing.assert_array_equal(
            processed_var.sensitivity({"p": 3, "q": 4}).full(),
            np.c_[y_sol.T, np.ones_like(y_sol).T],
        )

        # via value_and_sensitivity
        val, sens = processed_var.value_and_sensitivity({"p": 3, "q": 4})
        np.testing.assert_array_equal(val.full(), 3 * y_sol + 4)
        np.testing.assert_array_equal(
            sens.full(), np.c_[y_sol.T, np.ones_like(y_sol).T]
        )

        # Test bad inputs
        with self.assertRaisesRegex(TypeError, "inputs should be 'dict'"):
            processed_var.value(1)
        with self.assertRaisesRegex(ValueError, "Inconsistent input keys"):
            processed_var.value({"not p": 3})
        with self.assertRaisesRegex(ValueError, "Inconsistent input keys"):
            processed_var.value({"q": 3, "p": 2})

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
        processed_var = pybamm.ProcessedSymbolicVariable(var, solution)
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
        processed_eqn = pybamm.ProcessedSymbolicVariable(eqn_sol, sol)
        np.testing.assert_array_equal(
            processed_eqn.value(), y_sol + x_sol[:, np.newaxis]
        )

        # With vector t_sol
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)
        sol = pybamm.Solution(t_sol, y_sol)
        processed_eqn = pybamm.ProcessedSymbolicVariable(eqn_sol, sol)
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
        processed_eqn = pybamm.ProcessedSymbolicVariable(eqn_sol, sol)

        # Test values
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
        processed_eqn = pybamm.ProcessedSymbolicVariable(eqn_sol, sol)

        # Test values
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
        p = pybamm.InputParameter("p", domain=["negative electrode", "separator"])
        p.set_expected_size(65)
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
        processed_eqn = pybamm.ProcessedSymbolicVariable(eqn_sol, sol)

        # Test values - constant p
        np.testing.assert_array_equal(
            processed_eqn.value({"p": 27 * np.ones(n), "q": -42}),
            (27 * y_sol) ** 2 - 84,
        )
        # Test values - varying p
        p = np.linspace(0, 1, n)
        np.testing.assert_array_equal(
            processed_eqn.value({"p": p, "q": 3}), (p[:, np.newaxis] * y_sol) ** 2 + 6,
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

        # Bad shape
        with self.assertRaisesRegex(
            ValueError, "Wrong shape for input 'p': expected 65, actual 5"
        ):
            processed_eqn.value({"p": casadi.MX.sym("p", 5), "q": 1})

    def test_1D_different_domains(self):
        # Negative electrode domain
        var = pybamm.Variable("var", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)

        t_sol = [0]
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * 5
        sol = pybamm.Solution(t_sol, y_sol)
        pybamm.ProcessedSymbolicVariable(var_sol, sol)

        # Particle domain
        var = pybamm.Variable("var", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", domain=["negative particle"])

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(var)

        t_sol = [0]
        y_sol = np.ones_like(r_sol)[:, np.newaxis] * 5
        sol = pybamm.Solution(t_sol, y_sol)
        pybamm.ProcessedSymbolicVariable(var_sol, sol)

        # Current collector domain
        var = pybamm.Variable("var", domain=["current collector"])
        z = pybamm.SpatialVariable("z", domain=["current collector"])

        disc = tests.get_1p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        z_sol = disc.process_symbol(z).entries[:, 0]
        var_sol = disc.process_symbol(var)

        t_sol = [0]
        y_sol = np.ones_like(z_sol)[:, np.newaxis] * 5
        sol = pybamm.Solution(t_sol, y_sol)
        pybamm.ProcessedSymbolicVariable(var_sol, sol)

        # Other domain
        var = pybamm.Variable("var", domain=["line"])
        x = pybamm.SpatialVariable("x", domain=["line"])

        geometry = pybamm.Geometry(
            {
                "line": {
                    "primary": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
                }
            }
        )
        submesh_types = {"line": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
        var_pts = {x: 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, {"line": pybamm.FiniteVolume()})
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)

        t_sol = [0]
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * 5
        sol = pybamm.Solution(t_sol, y_sol)
        pybamm.ProcessedSymbolicVariable(var_sol, sol)

        # 2D fails
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )

        disc = tests.get_p2d_discretisation_for_testing()
        disc.set_variable_slices([var])
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(var)

        t_sol = [0]
        y_sol = np.ones_like(r_sol)[:, np.newaxis] * 5
        sol = pybamm.Solution(t_sol, y_sol)
        with self.assertRaisesRegex(NotImplementedError, "Shape not recognized"):
            pybamm.ProcessedSymbolicVariable(var_sol, sol)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
