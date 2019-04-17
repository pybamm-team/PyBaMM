#
# Test for the operator class
#
import pybamm
from tests import get_mesh_for_testing, get_p2d_mesh_for_testing

import numpy as np
from scipy.sparse import kron, eye
import unittest


class TestFiniteVolume(unittest.TestCase):
    def test_node_to_edge(self):
        a = pybamm.Symbol("a")

        def arithmetic_mean(array):
            return (array[1:] + array[:-1]) / 2

        ava = pybamm.Function(arithmetic_mean, a)
        self.assertEqual(ava.name, "function (arithmetic_mean)")
        self.assertEqual(ava.children[0].name, a.name)

        c = pybamm.Vector(np.ones(10))
        avc = pybamm.Function(arithmetic_mean, c)
        np.testing.assert_array_equal(avc.evaluate(), np.ones(9))

        d = pybamm.StateVector(slice(0, 10))
        y_test = np.ones(10)
        avd = pybamm.Function(arithmetic_mean, d)
        np.testing.assert_array_equal(avd.evaluate(None, y_test), np.ones(9))

    def test_extrapolate_left_right(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        macro_submesh = mesh.combine_submeshes(*whole_cell)
        micro_submesh = mesh["negative particle"]

        # Macroscale
        # create variable
        var = pybamm.Variable("var", domain=whole_cell)
        # boundary value should work with something more complicated than a variable
        extrap_left = pybamm.BoundaryValue(2 * var, "left")
        extrap_right = pybamm.BoundaryValue(4 - var, "right")
        disc.set_variable_slices([var])
        extrap_left_disc = disc.process_symbol(extrap_left)
        extrap_right_disc = disc.process_symbol(extrap_right)

        # check constant extrapolates to constant
        constant_y = np.ones_like(macro_submesh[0].nodes)
        self.assertEqual(extrap_left_disc.evaluate(None, constant_y), 2)
        self.assertEqual(extrap_right_disc.evaluate(None, constant_y), 3)

        # check linear variable extrapolates correctly
        linear_y = macro_submesh[0].nodes
        self.assertEqual(extrap_left_disc.evaluate(None, linear_y), 0)
        self.assertEqual(extrap_right_disc.evaluate(None, linear_y), 3)

        # Microscale
        # create variable
        var = pybamm.Variable("var", domain="negative particle")
        surf_eqn = pybamm.surf(var)
        disc.set_variable_slices([var])
        surf_eqn_disc = disc.process_symbol(surf_eqn)

        # check constant extrapolates to constant
        constant_y = np.ones_like(micro_submesh[0].nodes)
        self.assertEqual(surf_eqn_disc.evaluate(None, constant_y), 1)

        # check linear variable extrapolates correctly
        linear_y = micro_submesh[0].nodes
        y_surf = micro_submesh[0].nodes[-1] + micro_submesh[0].d_nodes[-1] / 2
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

        # test boundary conditions
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
            # Check that the equation can be evaluated in each case
            # Dirichlet
            disc._bcs = {var.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}}
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # Neumann
            disc._bcs = {flux.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(1)}}
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc._bcs = {
                var.id: {"left": pybamm.Scalar(0)},
                flux.id: {"right": pybamm.Scalar(1)},
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            disc._bcs = {
                flux.id: {"left": pybamm.Scalar(0)},
                var.id: {"right": pybamm.Scalar(1)},
            }
            eqn_disc = disc.process_symbol(eqn)
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

        # Test
        combined_submesh = mesh.combine_submeshes(*whole_cell)
        y_test = np.ones_like(combined_submesh[0].nodes)

        # left
        symbol_plus_ghost_left = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            var, discretised_symbol, lbc=lbc
        )
        np.testing.assert_array_equal(
            symbol_plus_ghost_left.evaluate(None, y_test)[1:],
            discretised_symbol.evaluate(None, y_test),
        )
        self.assertEqual(
            (
                symbol_plus_ghost_left.evaluate(None, y_test)[0]
                + symbol_plus_ghost_left.evaluate(None, y_test)[1]
            )
            / 2,
            0,
        )

        # right
        symbol_plus_ghost_right = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            var, discretised_symbol, rbc=rbc
        )
        np.testing.assert_array_equal(
            symbol_plus_ghost_right.evaluate(None, y_test)[:-1],
            discretised_symbol.evaluate(None, y_test),
        )
        self.assertEqual(
            (
                symbol_plus_ghost_right.evaluate(None, y_test)[-2]
                + symbol_plus_ghost_right.evaluate(None, y_test)[-1]
            )
            / 2,
            3,
        )

        # both
        symbol_plus_ghost_both = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            var, discretised_symbol, lbc, rbc
        )
        np.testing.assert_array_equal(
            symbol_plus_ghost_both.evaluate(None, y_test)[1:-1],
            discretised_symbol.evaluate(None, y_test),
        )
        self.assertEqual(
            (
                symbol_plus_ghost_both.evaluate(None, y_test)[0]
                + symbol_plus_ghost_both.evaluate(None, y_test)[1]
            )
            / 2,
            0,
        )
        self.assertEqual(
            (
                symbol_plus_ghost_both.evaluate(None, y_test)[-2]
                + symbol_plus_ghost_both.evaluate(None, y_test)[-1]
            )
            / 2,
            3,
        )

        # test errors
        with self.assertRaisesRegex(ValueError, "at least one"):
            pybamm.FiniteVolume(mesh).add_ghost_nodes(
                var, discretised_symbol, None, None
            )

        with self.assertRaisesRegex(
            TypeError, "discretised_symbol must be a StateVector or Concatenation"
        ):
            pybamm.FiniteVolume(mesh).add_ghost_nodes(var, pybamm.Scalar(1), None, None)

    def test_add_ghost_nodes_concatenation(self):
        # Set up

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Add ghost nodes
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var_n = pybamm.Variable("var", domain=["negative electrode"])
        var_s = pybamm.Variable("var", domain=["separator"])
        var_p = pybamm.Variable("var", domain=["positive electrode"])
        var = pybamm.Concatenation(var_n, var_s, var_p)
        disc.set_variable_slices([var])
        discretised_symbol = disc.process_symbol(var)
        lbc = pybamm.Scalar(0)
        rbc = pybamm.Scalar(3)

        # Test
        combined_submesh = mesh.combine_submeshes(*whole_cell)
        y_test = np.ones_like(combined_submesh[0].nodes)

        # both
        symbol_plus_ghost_both = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            var, discretised_symbol, lbc, rbc
        )
        np.testing.assert_array_equal(
            symbol_plus_ghost_both.evaluate(None, y_test)[1:-1],
            discretised_symbol.evaluate(None, y_test),
        )
        self.assertEqual(
            (
                symbol_plus_ghost_both.evaluate(None, y_test)[0]
                + symbol_plus_ghost_both.evaluate(None, y_test)[1]
            )
            / 2,
            0,
        )
        self.assertEqual(
            (
                symbol_plus_ghost_both.evaluate(None, y_test)[-2]
                + symbol_plus_ghost_both.evaluate(None, y_test)[-1]
            )
            / 2,
            3,
        )

    def test_p2d_add_ghost_nodes(self):
        # create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # add ghost nodes
        c_s_n = pybamm.Variable("c_s_n", domain=["negative particle"])
        c_s_p = pybamm.Variable("c_s_p", domain=["positive particle"])

        disc.set_variable_slices([c_s_n])
        disc_c_s_n = pybamm.StateVector(disc._y_slices[c_s_n.id])

        disc.set_variable_slices([c_s_p])
        disc_c_s_p = pybamm.StateVector(disc._y_slices[c_s_p.id])
        lbc = pybamm.Scalar(0)
        rbc = pybamm.Scalar(3)
        c_s_n_plus_ghost = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            c_s_n, disc_c_s_n, lbc, rbc
        )
        c_s_p_plus_ghost = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            c_s_p, disc_c_s_p, lbc, rbc
        )

        mesh_s_n = mesh["negative particle"]
        mesh_s_p = mesh["positive particle"]

        n_prim_pts = mesh_s_n[0].npts
        n_sec_pts = len(mesh_s_n)

        p_prim_pts = mesh_s_p[0].npts
        p_sec_pts = len(mesh_s_p)

        y_s_n_test = np.kron(np.ones(n_sec_pts), np.ones(n_prim_pts))
        y_s_p_test = np.kron(np.ones(p_sec_pts), np.ones(p_prim_pts))

        # evaluate with and without ghost points
        c_s_n_eval = disc_c_s_n.evaluate(None, y_s_n_test)
        c_s_n_ghost_eval = c_s_n_plus_ghost.evaluate(None, y_s_n_test)

        c_s_p_eval = disc_c_s_p.evaluate(None, y_s_p_test)
        c_s_p_ghost_eval = c_s_p_plus_ghost.evaluate(None, y_s_p_test)

        # reshape to make easy to deal with
        c_s_n_eval = np.reshape(c_s_n_eval, [n_sec_pts, n_prim_pts])
        c_s_n_ghost_eval = np.reshape(c_s_n_ghost_eval, [n_sec_pts, n_prim_pts + 2])

        c_s_p_eval = np.reshape(c_s_p_eval, [p_sec_pts, p_prim_pts])
        c_s_p_ghost_eval = np.reshape(c_s_p_ghost_eval, [p_sec_pts, p_prim_pts + 2])

        np.testing.assert_array_equal(c_s_n_ghost_eval[:, 1:-1], c_s_n_eval)
        np.testing.assert_array_equal(c_s_p_ghost_eval[:, 1:-1], c_s_p_eval)

        np.testing.assert_array_equal(
            (c_s_n_ghost_eval[:, 0] + c_s_n_ghost_eval[:, 1]) / 2, 0
        )
        np.testing.assert_array_equal(
            (c_s_p_ghost_eval[:, 0] + c_s_p_ghost_eval[:, 1]) / 2, 0
        )

        np.testing.assert_array_equal(
            (c_s_n_ghost_eval[:, -2] + c_s_n_ghost_eval[:, -1]) / 2, 3
        )
        np.testing.assert_array_equal(
            (c_s_p_ghost_eval[:, -2] + c_s_p_ghost_eval[:, -1]) / 2, 3
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

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        disc._bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        # test ghost cells
        self.assertTrue(grad_eqn_disc.children[1].has_left_ghost_cell)
        self.assertTrue(grad_eqn_disc.children[1].has_right_ghost_cell)

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

    def test_p2d_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        in the pseudo 2-dimensional case
        """

        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n_mesh = mesh["negative particle"]

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        var = pybamm.Variable("var", domain=["negative particle"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        disc._bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        prim_pts = n_mesh[0].npts
        sec_pts = len(n_mesh)
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))

        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])

        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts + 1]))

        # div: test on linear r^2
        # div (grad r^2) = 6
        const = 6 * np.ones(sec_pts * prim_pts)

        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(6), "right": pybamm.Scalar(6)}
        }
        disc._bcs = boundary_conditions

        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, const)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_array_almost_equal(div_eval, np.zeros([sec_pts, prim_pts]))

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

    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on c) on
        one side and Neumann boundary conditions (applied by div on N) on the other
        """
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

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {"left": pybamm.Scalar(1)},
            N.id: {"right": pybamm.Scalar(0)},
        }
        disc._bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # test ghost cells
        self.assertTrue(grad_eqn_disc.children[1].has_left_ghost_cell)
        self.assertFalse(grad_eqn_disc.children[1].has_right_ghost_cell)

        # Constant y should have gradient and laplacian zero
        constant_y = np.ones_like(combined_submesh[0].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[:-1]),
        )
        np.testing.assert_array_equal(
            div_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].nodes),
        )

        boundary_conditions = {
            var.id: {"right": pybamm.Scalar(1)},
            N.id: {"left": pybamm.Scalar(1)},
        }
        disc._bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # test ghost cells
        self.assertFalse(grad_eqn_disc.children[1].has_left_ghost_cell)
        self.assertTrue(grad_eqn_disc.children[1].has_right_ghost_cell)

        # Linear y should have gradient one and laplacian zero
        linear_y = combined_submesh[0].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[:-1]),
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

    def test_p2d_spherical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        in the pseudo 2-dimensional case
        """

        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n_mesh = mesh["negative particle"]

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # test grad
        var = pybamm.Variable("var", domain=["negative particle"])
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        prim_pts = n_mesh[0].npts
        sec_pts = len(n_mesh)
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))

        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts - 1])

        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts - 1]))

        # div
        # div (grad r^2) = 6, N_left = N_right = 0
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        disc._bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        const = 6 * np.ones(sec_pts * prim_pts)
        div_eval = div_eqn_disc.evaluate(None, const)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_array_almost_equal(div_eval, np.zeros([sec_pts, prim_pts]))

    def test_grad_div_shapes_mixed_domain(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

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

    def test_indefinite_integral(self):

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # input a phi, take grad, then integrate to recover phi approximation
        # (need to test this way as check evaluated on edges using if has grad
        # and no div)
        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        i = pybamm.grad(phi)  # create test current (variable on edges)

        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        int_grad_phi = pybamm.IndefiniteIntegral(i, x)
        disc.set_variable_slices([phi])  # i is not a fundamental variable

        int_grad_phi_disc = disc.process_symbol(int_grad_phi)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")

        # constant case
        phi_exact = np.ones_like(combined_submesh[0].nodes)
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)

        # linear case
        phi_exact = combined_submesh[0].nodes
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += phi_exact[0]  # add constant of integration
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)

        # sine case
        phi_exact = np.sin(combined_submesh[0].nodes)
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += phi_exact[0]  # add constant of integration
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)

        # --------------------------------------------------------------------
        # region which doesn't start at zero
        phi = pybamm.Variable("phi", domain=["separator", "positive electrode"])
        i = pybamm.grad(phi)  # create test current (variable on edges)
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        int_grad_phi = pybamm.IndefiniteIntegral(i, x)
        disc.set_variable_slices([phi])  # i is not a fundamental variable

        int_grad_phi_disc = disc.process_symbol(int_grad_phi)
        combined_submesh = mesh.combine_submeshes("separator", "positive electrode")

        # constant case
        phi_exact = np.ones_like(combined_submesh[0].nodes)
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)

        # linear case
        phi_exact = combined_submesh[0].nodes
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += phi_exact[0]  # add constant of integration
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)

        # sine case
        phi_exact = np.sin(combined_submesh[0].nodes)
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += phi_exact[0]  # add constant of integration
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)

        # --------------------------------------------------------------------
        # micrsoscale case
        c = pybamm.Variable("c", domain=["negative particle"])
        N = pybamm.grad(c)  # create test current (variable on edges)
        r_n = pybamm.SpatialVariable("r_n", ["negative particle"])
        c_integral = pybamm.IndefiniteIntegral(N, r_n)
        disc.set_variable_slices([c])  # N is not a fundamental variable

        c_integral_disc = disc.process_symbol(c_integral)
        combined_submesh = mesh["negative particle"]

        # constant case
        c_exact = np.ones_like(combined_submesh[0].nodes)
        c_approx = c_integral_disc.evaluate(None, c_exact)
        c_approx += 1  # add constant of integration
        np.testing.assert_array_equal(c_exact, c_approx)

        # linear case
        c_exact = combined_submesh[0].nodes
        c_approx = c_integral_disc.evaluate(None, c_exact)
        c_approx += c_exact[0]  # add constant of integration
        np.testing.assert_array_almost_equal(c_exact, c_approx)

        # sine case
        c_exact = np.sin(combined_submesh[0].nodes)
        c_approx = c_integral_disc.evaluate(None, c_exact)
        c_approx += c_exact[0]  # add constant of integration
        np.testing.assert_array_almost_equal(c_exact, c_approx)

        # ------------
        # check raises error for variales not on mesh edges
        phi = pybamm.Variable("phi", domain=["separator", "positive electrode"])
        no_grad_or_div = phi
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        int_grad_phi = pybamm.IndefiniteIntegral(no_grad_or_div, x)
        disc.set_variable_slices([phi])

        with self.assertRaisesRegex(pybamm.ModelError, "integrated"):
            disc.process_symbol(int_grad_phi)

        grad_and_div = pybamm.div(pybamm.grad(phi))
        int_grad_phi = pybamm.IndefiniteIntegral(grad_and_div, x)
        disc.set_variable_slices([phi])

        with self.assertRaisesRegex(pybamm.ModelError, "integrated"):
            disc.process_symbol(int_grad_phi)

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

    def test_mass_matrix_shape(self):
        """
        Test mass matrix shape
        """
        # one equation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        model.variables = {"c": c, "N": N}

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)
        disc.process_model(model)

        # mass matrix
        mass = np.eye(combined_submesh[0].npts)
        np.testing.assert_array_equal(mass, model.mass_matrix.entries.toarray())

    def test_p2d_mass_matrix_shape(self):
        """
        Test mass matrix shape in the pseudo 2-dimensional case
        """
        c = pybamm.Variable("c", domain=["negative particle"])
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        model.variables = {"c": c, "N": N}
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        prim_pts = mesh["negative particle"][0].npts
        sec_pts = len(mesh["negative particle"])
        mass_local = eye(prim_pts)
        mass = kron(eye(sec_pts), mass_local)
        np.testing.assert_array_equal(
            mass.toarray(), model.mass_matrix.entries.toarray()
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
