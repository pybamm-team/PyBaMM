#
# Test for the operator class
#
import pybamm

import numpy as np
import unittest


class TestFiniteVolumeDiscretisation(unittest.TestCase):
    def test_node_to_edge(self):
        a = pybamm.Symbol("a")

        def arithmetic_mean(array):
            return (array[1:] + array[:-1]) / 2

        ava = pybamm.NodeToEdge(a, arithmetic_mean)
        self.assertEqual(ava.name, "node to edge (arithmetic_mean)")
        self.assertEqual(ava.children[0].name, a.name)

        b = pybamm.Scalar(-4)
        avb = pybamm.NodeToEdge(b, arithmetic_mean)
        self.assertEqual(avb.evaluate(), -4)

        c = pybamm.Vector(np.ones(10))
        avc = pybamm.NodeToEdge(c, arithmetic_mean)
        np.testing.assert_array_equal(avc.evaluate(), np.ones(9))

        d = pybamm.StateVector(slice(0, 10))
        y_test = np.ones(10)
        avd = pybamm.NodeToEdge(d, arithmetic_mean)
        np.testing.assert_array_equal(avd.evaluate(None, y_test), np.ones(9))

    def test_discretise_diffusivity_times_spatial_operator(self):
        # Set up
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        # Discretise some equations where averaging is needed
        var = pybamm.Variable("var", domain=["whole cell"])
        y_slices = disc.get_variable_slices([var])
        y_test = np.ones_like(mesh["whole cell"].nodes)
        for eqn in [
            var * pybamm.grad(var),
            var ** 2 * pybamm.grad(var),
            var * pybamm.grad(var) ** 2,
            var * (pybamm.grad(var) + 2),
            (pybamm.grad(var) + 2) * (-var),
            (pybamm.grad(var) + 2) * (2 * var),
            pybamm.grad(var) * pybamm.grad(var),
            (pybamm.grad(var) + 2) * pybamm.grad(var) ** 2,
        ]:
            eqn_disc = disc.process_symbol(eqn, y_slices, {})
            # Check that the equation can be evaluated
            eqn_disc.evaluate(None, y_test)

        # more testing, with boundary conditions
        for flux, eqn in zip(
            [
                pybamm.grad(var),
                pybamm.grad(var),
                pybamm.grad(var),
                2 * pybamm.grad(var),
                2 * pybamm.grad(var),
                var * pybamm.grad(var) + 2 * pybamm.grad(var),
            ],
            [
                pybamm.div(pybamm.grad(var)),
                pybamm.div(pybamm.grad(var)) + 2,
                pybamm.div(pybamm.grad(var)) + var,
                pybamm.div(2 * pybamm.grad(var)),
                pybamm.div(2 * pybamm.grad(var)) + 3 * var,
                -2 * pybamm.div(var * pybamm.grad(var) + 2 * pybamm.grad(var)),
            ],
        ):
            eqn_disc = disc.process_symbol(
                eqn,
                y_slices,
                {flux.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}},
            )
            # Check that the equation can be evaluated
            eqn_disc.evaluate(None, y_test)

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

    def test_grad_div_shapes_mixed_domain(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        # grad
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        y_slices = disc.get_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh.nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": pybamm.Scalar(0),
                "right": pybamm.Scalar(combined_submesh.edges[-1]),
            }
        }

        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y), np.ones_like(combined_submesh.edges)
        )

        div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y), np.zeros_like(combined_submesh.nodes)
        )

    def test_integral(self):
        """
        Test integral
        """
        Ln = 0.1
        Ls = 0.2
        Lp = 0.3
        L = Ln + Ls + Lp
        param = pybamm.ParameterValues(base_parameters={"Ln": Ln, "Ls": Ls, "Lp": Lp})
        mesh = pybamm.FiniteVolumeMacroMesh(param, 200)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        integral_eqn = pybamm.Integral(
            var, pybamm.Space(["negative electrode", "separator"])
        )
        y_slices = disc.get_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn, y_slices)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh.nodes)
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), (Ln + Ls) / L)
        linear_y = combined_submesh.nodes
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, linear_y), ((Ln + Ls) / L) ** 2 / 2
        )
        cos_y = np.cos(combined_submesh.nodes)
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, cos_y), np.sin((Ln + Ls) / L)
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
