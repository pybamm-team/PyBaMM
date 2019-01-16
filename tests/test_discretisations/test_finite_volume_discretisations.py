#
# Test for the operator class
#
import pybamm

import numpy as np
import unittest


class TestFiniteVolumeDiscretisation(unittest.TestCase):
    def test_add_ghost_nodes(self):
        # Set up
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        # Add ghost nodes
        var = pybamm.Variable("var", domain=["whole cell"])
        y_slices = disc.get_variable_slices([var])
        discretised_symbol = pybamm.StateVector(y_slices[var.id])
        lbc = pybamm.Scalar(0)
        rbc = pybamm.Scalar(3)
        symbol_plus_ghost = disc.add_ghost_nodes(discretised_symbol, lbc, rbc)

        # Test
        y_test = np.ones_like(mesh["whole cell"].nodes)
        np.testing.assert_array_equal(
            symbol_plus_ghost.evaluate(None, y_test)[1:-1],
            discretised_symbol.evaluate(None, y_test),
        )
        self.assertEqual(
            (
                symbol_plus_ghost.evaluate(None, y_test)[0]
                + symbol_plus_ghost.evaluate(None, y_test)[1]
            )
            / 2,
            0,
        )
        self.assertEqual(
            (
                symbol_plus_ghost.evaluate(None, y_test)[-2]
                + symbol_plus_ghost.evaluate(None, y_test)[-1]
            )
            / 2,
            3,
        )

    def test_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        # grad
        var = pybamm.Variable("var", domain=["whole cell"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        y_slices = disc.get_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)

        constant_y = np.ones_like(mesh["whole cell"].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(mesh["whole cell"].edges),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = mesh["whole cell"].nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}
        }

        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(mesh["whole cell"].edges),
        )

        div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(mesh["whole cell"].nodes),
        )

    def test_grad_div_shapes_Neumann_bcs(self):
        """Test grad and div with Neumann boundary conditions (applied by div on N)"""
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        # grad
        var = pybamm.Variable("var", domain=["whole cell"])
        grad_eqn = pybamm.grad(var)
        y_slices = disc.get_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, {})

        constant_y = np.ones_like(mesh["whole cell"].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(mesh["whole cell"].edges[1:-1]),
        )

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)

        # Linear y should have laplacian zero
        linear_y = mesh["whole cell"].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(mesh["whole cell"].edges[1:-1]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(mesh["whole cell"].nodes),
        )

    def test_grad_convergence_without_bcs(self):
        # Convergence
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        var = pybamm.Variable("var", domain=["whole cell"])
        grad_eqn = pybamm.grad(var)

        # Function for convergence testing
        def get_l2_error(n):
            # Set up discretisation
            mesh = pybamm.FiniteVolumeMacroMesh(param, target_npts=n)
            disc = pybamm.FiniteVolumeDiscretisation(mesh)

            # Define exact solutions
            y = np.sin(mesh["whole cell"].nodes)
            grad_exact = np.cos(mesh["whole cell"].edges[1:-1])

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, {})
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Calculate errors
            return np.linalg.norm(grad_approx - grad_exact) / np.linalg.norm(grad_exact)

        # Get errors
        ns = 100 * (2 ** np.arange(1, 7))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**2 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_grad_convergence_with_bcs(self):
        # Convergence
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        var = pybamm.Variable("var", domain=["whole cell"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": pybamm.Scalar(np.sin(0)),
                "right": pybamm.Scalar(np.sin(1)),
            }
        }

        # Function for convergence testing
        def get_l2_error(n):
            # Set up discretisation
            mesh = pybamm.FiniteVolumeMacroMesh(param, target_npts=n)
            disc = pybamm.FiniteVolumeDiscretisation(mesh)

            # Define exact solutions
            y = np.sin(mesh["whole cell"].nodes)
            grad_exact = np.cos(mesh["whole cell"].edges)

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Calculate errors
            return np.linalg.norm(grad_approx - grad_exact) / np.linalg.norm(grad_exact)

        # Get errors
        ns = 100 * (2 ** np.arange(1, 7))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.49 * np.ones_like(rates), rates)

    def test_div_convergence_internal(self):
        # Convergence
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        var = pybamm.Variable("var", domain=["whole cell"])
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        # Function for convergence testing
        def get_l2_error(n):
            # Set up discretisation
            mesh = pybamm.FiniteVolumeMacroMesh(param, target_npts=n)
            disc = pybamm.FiniteVolumeDiscretisation(mesh)

            # Define exact solutions
            y = np.sin(mesh["whole cell"].nodes)
            div_exact_internal = -np.sin(mesh["whole cell"].nodes[1:-1])

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
            div_approx_internal = div_eqn_disc.evaluate(None, y)[1:-1]

            # Calculate errors
            return np.linalg.norm(
                div_approx_internal - div_exact_internal
            ) / np.linalg.norm(div_exact_internal)

        # Get errors
        ns = 10 * (2 ** np.arange(1, 6))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**2 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_div_convergence(self):
        # Convergence
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        var = pybamm.Variable("var", domain=["whole cell"])
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        # Function for convergence testing
        def get_l2_error(n):
            # Set up discretisation
            mesh = pybamm.FiniteVolumeMacroMesh(param, target_npts=n)
            disc = pybamm.FiniteVolumeDiscretisation(mesh)

            # Define exact solutions
            y = np.sin(mesh["whole cell"].nodes)
            div_exact_internal = -np.sin(mesh["whole cell"].nodes[1:-1])

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
            div_approx_internal = div_eqn_disc.evaluate(None, y)[1:-1]

            # Calculate errors
            return np.linalg.norm(
                div_approx_internal - div_exact_internal
            ) / np.linalg.norm(div_exact_internal)

        # Get errors
        ns = 10 * (2 ** np.arange(1, 6))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence because of boundary conditions
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.49 * np.ones_like(rates), rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
