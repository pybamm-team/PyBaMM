#
# Test for the operator class
#
import pybamm
from tests import get_mesh_for_testing

import numpy as np
import unittest


class TestFiniteVolume(unittest.TestCase):
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

    def test_surface_value(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes("negative particle")

        # create variable
        var = pybamm.Variable("var", domain="negative particle")
        surf_eqn = pybamm.surf(var)
        disc.set_variable_slices([var])
        surf_eqn_disc = disc.process_symbol(surf_eqn)

        # check constant extrapolates to constant
        constant_y = np.ones_like(combined_submesh[0].nodes)
        self.assertEqual(surf_eqn_disc.evaluate(None, constant_y), 1)

        # check linear variable extrapolates correctly
        linear_y = combined_submesh[0].nodes
        y_surf = combined_submesh[0].nodes[-1] + combined_submesh[0].d_nodes[-1] / 2
        self.assertEqual(surf_eqn_disc.evaluate(None, linear_y), y_surf)

    def test_discretise_diffusivity_times_spatial_operator(self):
        # Set up
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # Discretise some equations where averaging is needed
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y_test = np.ones_like(combined_submesh[0].nodes)
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
            eqn_disc = disc.process_symbol(eqn)
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
            disc._bcs = {flux.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}}

            eqn_disc = disc.process_symbol(eqn)
            # Check that the equation can be evaluated
            eqn_disc.evaluate(None, y_test)

    def test_add_ghost_nodes(self):
        # Set up

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Add ghost nodes
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        discretised_symbol = pybamm.StateVector(disc._y_slices[var.id])
        lbc = pybamm.Scalar(0)
        rbc = pybamm.Scalar(3)
        symbol_plus_ghost = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            var, discretised_symbol, lbc, rbc
        )

        # Test
        combined_submesh = mesh.combine_submeshes(*whole_cell)
        y_test = np.ones_like(combined_submesh[0].nodes)
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
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        disc._bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh[0].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh[0].nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}
        }

        disc._bcs = boundary_conditions

        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh[0].nodes),
        )

    def test_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes("negative particle")

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # grad
        # grad(r) == 1
        var = pybamm.Variable("var", domain=["negative particle"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }

        disc._bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh[0].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges),
        )

        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}
        }
        disc._bcs = boundary_conditions

        y_linear = combined_submesh[0].nodes
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, y_linear),
            np.ones_like(combined_submesh[0].edges),
        )

        # div: test on linear r^2
        # div (grad r^2) = 6
        const = 6 * np.ones(combined_submesh[0].npts)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(6), "right": pybamm.Scalar(6)}
        }
        disc._bcs = boundary_conditions

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, const), np.zeros_like(combined_submesh[0].nodes)
        )

    def test_grad_div_shapes_Neumann_bcs(self):
        """Test grad and div with Neumann boundary conditions (applied by div on N)"""
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh[0].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[1:-1]),
        )

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        disc._bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Linear y should have laplacian zero
        linear_y = combined_submesh[0].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[1:-1]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh[0].nodes),
        )

    def test_spherical_grad_div_shapes_Neumann_bcs(self):
        """Test grad and div with Neumann boundary conditions (applied by div on N)"""

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes("negative particle")

        # grad
        var = pybamm.Variable("var", domain="negative particle")
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh[0].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[1:-1]),
        )

        linear_y = combined_submesh[0].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[1:-1]),
        )
        # div
        # div ( grad(r^2) ) == 6 , N_left = N_right = 0
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        disc._bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        linear_y = combined_submesh[0].nodes
        const = 6 * np.ones(combined_submesh[0].npts)

        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, const), np.zeros_like(combined_submesh[0].nodes)
        )

    def test_grad_div_shapes_mixed_domain(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # grad
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        disc._bcs = boundary_conditions

        disc.set_variable_slices([var])

        grad_eqn_disc = disc.process_symbol(grad_eqn)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh[0].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh[0].nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": pybamm.Scalar(0),
                "right": pybamm.Scalar(combined_submesh[0].edges[-1]),
            }
        }
        disc._bcs = boundary_conditions

        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh[0].nodes),
        )

    def test_definite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing(200)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"][0].edges[-1]
        ls = mesh["separator"][0].edges[-1] - ln
        lp = 1 - (ln + ls)

        # macroscale variable
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh[0].nodes)
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ln + ls)
        linear_y = combined_submesh[0].nodes
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, linear_y), (ln + ls) ** 2 / 2
        )
        cos_y = np.cos(combined_submesh[0].nodes)
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, cos_y), np.sin(ln + ls), places=4
        )

        # domain not starting at zero
        var = pybamm.Variable("var", domain=["separator", "positive electrode"])
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        combined_submesh = mesh.combine_submeshes("separator", "positive electrode")
        constant_y = np.ones_like(combined_submesh[0].nodes)
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ls + lp)
        linear_y = combined_submesh[0].nodes
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, linear_y), (1 - (ln) ** 2) / 2
        )
        cos_y = np.cos(combined_submesh[0].nodes)
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, cos_y), np.sin(1) - np.sin(ln), places=4
        )

        # microscale variable
        var = pybamm.Variable("var", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", ["negative particle"])
        integral_eqn = pybamm.Integral(var, r)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        constant_y = np.ones_like(mesh["negative particle"][0].nodes)
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), np.pi)
        linear_y = mesh["negative particle"][0].nodes
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, linear_y), 2 * np.pi / 3, places=4
        )
        one_over_y = 1 / mesh["negative particle"][0].nodes
        self.assertEqual(integral_eqn_disc.evaluate(None, one_over_y), 2 * np.pi)

    @unittest.skip("indefinite integral not yet implemented")
    def test_indefinite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges[-1]

        # macroscale variable
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        integral_eqn = pybamm.IndefiniteIntegral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh.nodes)
        constant_y_edges = np.ones_like(combined_submesh.edges)
        linear_y = combined_submesh.nodes
        linear_y_edges = combined_submesh.edges
        np.testing.assert_array_equal(
            integral_eqn_disc.evaluate(None, constant_y_edges), linear_y
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y_edges), linear_y ** 2 / 2
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, np.cos(linear_y_edges)),
            np.sin(linear_y),
            places=4,
        )

        # domain not starting at zero
        var = pybamm.Variable("var", domain=["separator", "positive electrode"])
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        integral_eqn = pybamm.IndefiniteIntegral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        combined_submesh = mesh.combine_submeshes("separator", "positive electrode")
        np.testing.assert_array_equal(
            integral_eqn_disc.evaluate(None, constant_y), linear_y - ln
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), (linear_y ** 2 - (ln) ** 2) / 2
        )
        cos_y = np.cos(combined_submesh.nodes)
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y),
            np.sin(linear_y) - np.sin(ln),
            places=4,
        )

        # microscale variable
        var = pybamm.Variable("var", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", ["negative particle"])
        integral_eqn = pybamm.IndefiniteIntegral(var, r)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        constant_y = np.ones_like(mesh["negative particle"].nodes)
        np.testing.assert_array_equal(
            integral_eqn_disc.evaluate(None, constant_y), np.pi
        )
        linear_y = mesh["negative particle"].nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), 2 * np.pi / 3, places=4
        )
        one_over_y = 1 / mesh["negative particle"].nodes
        np.testing.assert_array_equal(
            integral_eqn_disc.evaluate(None, one_over_y), 2 * np.pi
        )

    def test_grad_convergence_without_bcs(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)

        # Function for convergence testing
        def get_l2_error(n):
            # Set up discretisation
            n = 3 * round(n / 3)
            # create discretisation
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"macroscale": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)

            combined_submesh = mesh.combine_submeshes(*whole_cell)
            # Define exact solutions
            y = np.sin(combined_submesh[0].nodes)
            grad_exact = np.cos(combined_submesh[0].edges[1:-1])

            # Discretise and evaluate
            disc.set_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn)
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Calculate errors
            return np.linalg.norm(grad_approx - grad_exact) / np.linalg.norm(grad_exact)

        # Get errors
        ns = 100 * (2 ** np.arange(2, 7))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**2 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_grad_convergence_with_bcs(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": pybamm.Scalar(np.sin(0)),
                "right": pybamm.Scalar(np.sin(1)),
            }
        }

        # Function for convergence testing
        def get_l2_error(n):
            # create discretisation
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"macroscale": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions

            # Set up discretisation
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            combined_submesh = mesh.combine_submeshes(*whole_cell)

            mesh.add_ghost_meshes()
            disc.mesh.add_ghost_meshes()

            # Define exact solutions
            y = np.sin(combined_submesh[0].nodes)
            grad_exact = np.cos(combined_submesh[0].edges)

            # Discretise and evaluate
            disc.set_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn)
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Calculate errors
            return np.linalg.norm(grad_approx - grad_exact) / np.linalg.norm(grad_exact)

        # Get errors
        ns = 100 * (2 ** np.arange(2, 7))
        ns = 3 * np.round(ns / 3)
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.49 * np.ones_like(rates), rates)

    def test_div_convergence_internal(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        # Function for convergence testing
        def get_l2_error(n):

            # create discretisation
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"macroscale": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions

            whole_cell = ["negative electrode", "separator", "positive electrode"]
            combined_submesh = mesh.combine_submeshes(*whole_cell)

            # Define exact solutions
            y = np.sin(combined_submesh[0].nodes)
            div_exact_internal = -np.sin(combined_submesh[0].nodes[1:-1])

            # Discretise and evaluate
            disc.set_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx_internal = div_eqn_disc.evaluate(None, y)[1:-1]

            # Calculate errors
            return np.linalg.norm(
                div_approx_internal - div_exact_internal
            ) / np.linalg.norm(div_exact_internal)

        # Get errors
        ns = 10 * (2 ** np.arange(2, 6))
        ns = 3 * np.round(ns / 3)
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**2 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.9 * np.ones_like(rates), rates)

    def test_div_convergence(self):
        # Convergence
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        # Function for convergence testing
        def get_l2_error(n):
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            # create discretisation
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"macroscale": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions

            combined_submesh = mesh.combine_submeshes(*whole_cell)

            # Define exact solutions
            y = np.sin(combined_submesh[0].nodes)
            div_exact_internal = -np.sin(combined_submesh[0].nodes[1:-1])

            # Discretise and evaluate
            disc.set_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn)
            div_approx_internal = div_eqn_disc.evaluate(None, y)[1:-1]

            # Calculate errors
            return np.linalg.norm(
                div_approx_internal - div_exact_internal
            ) / np.linalg.norm(div_exact_internal)

        # Get errors
        ns = 10 * (2 ** np.arange(2, 6))
        ns = 3 * np.round(ns / 3)
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence because of boundary conditions
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.49 * np.ones_like(rates), rates)

    def test_spherical_operators(self):
        # test div( grad( sin(r) )) == (2/r)*cos(r) - *sin(r)

        domain = ["negative particle"]
        c = pybamm.Variable("c", domain=domain)
        N = pybamm.grad(c)
        eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        def get_l2_error(n):
            mesh = get_mesh_for_testing(n)
            spatial_methods = {"negative particle": pybamm.FiniteVolume}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc._bcs = boundary_conditions
            mesh = disc.mesh["negative particle"]
            r = mesh[0].nodes

            # exact solution
            y = np.sin(r)
            exact = (2 / r) * np.cos(r) - np.sin(r)
            exact_internal = exact[1:-1]

            # discretise and evaluate
            variables = [c]
            disc.set_variable_slices(variables)
            eqn_disc = disc.process_symbol(eqn)
            approx_internal = eqn_disc.evaluate(None, y)[1:-1]

            # error
            error = np.linalg.norm(approx_internal - exact_internal) / np.linalg.norm(
                exact_internal
            )
            return error

        # Get errors
        ns = 10 * (2 ** np.arange(2, 7))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**1.5 convergence because of boundary conditions
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)

    def test_discretise_spatial_variable(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # space
        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x1_disc = disc.process_symbol(x1)
        self.assertIsInstance(x1_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x1_disc.evaluate(), disc.mesh["negative electrode"][0].nodes
        )

        z = pybamm.SpatialVariable("z", ["negative electrode"])
        with self.assertRaises(NotImplementedError):
            disc.process_symbol(z)

        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x2_disc = disc.process_symbol(x2)
        self.assertIsInstance(x2_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x2_disc.evaluate(),
            disc.mesh.combine_submeshes("negative electrode", "separator")[0].nodes,
        )

        r = 3 * pybamm.SpatialVariable("r", ["negative particle"])
        r_disc = disc.process_symbol(r)
        self.assertIsInstance(r_disc.children[1], pybamm.Vector)
        np.testing.assert_array_equal(
            r_disc.evaluate(), 3 * disc.mesh["negative particle"][0].nodes
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
