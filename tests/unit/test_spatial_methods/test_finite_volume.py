#
# Test for the operator class
#
import pybamm
from tests import (
    get_mesh_for_testing,
    get_p2d_mesh_for_testing,
    get_1p1d_mesh_for_testing,
)

import numpy as np
from scipy.sparse import kron, eye
import unittest


class TestFiniteVolume(unittest.TestCase):
    def test_node_to_edge_to_node(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume(mesh)
        n = mesh["negative electrode"][0].npts

        # node to edge
        c = pybamm.Vector(np.ones(n), domain=["negative electrode"])
        diffusivity_c = fin_vol.node_to_edge(c)
        np.testing.assert_array_equal(diffusivity_c.evaluate(), np.ones((n + 1, 1)))

        # edge to node
        d = pybamm.StateVector(slice(0, n + 1), domain=["negative electrode"])
        y_test = np.ones(n + 1)
        diffusivity_d = fin_vol.edge_to_node(d)
        np.testing.assert_array_equal(
            diffusivity_d.evaluate(None, y_test), np.ones((n, 1))
        )

        # bad shift key
        with self.assertRaisesRegex(ValueError, "shift key"):
            fin_vol.shift(c, "bad shift key")

    def test_concatenation(self):
        mesh = get_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume(mesh)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        edges = [pybamm.Vector(mesh[dom][0].edges, domain=dom) for dom in whole_cell]
        # Concatenation of edges should get averaged to nodes first, using edge_to_node
        v_disc = fin_vol.concatenation(edges)
        np.testing.assert_array_equal(
            v_disc.evaluate()[:, 0], mesh.combine_submeshes(*whole_cell)[0].nodes
        )

        # test for bad shape
        edges = [
            pybamm.Vector(np.ones(mesh[dom][0].npts + 2), domain=dom)
            for dom in whole_cell
        ]
        with self.assertRaisesRegex(pybamm.ShapeError, "child must have size n_nodes"):
            fin_vol.concatenation(edges)

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
        constant_y = np.ones_like(macro_submesh[0].nodes[:, np.newaxis])
        self.assertEqual(extrap_left_disc.evaluate(None, constant_y), 2)
        self.assertEqual(extrap_right_disc.evaluate(None, constant_y), 3)

        # check linear variable extrapolates correctly
        linear_y = macro_submesh[0].nodes
        self.assertEqual(extrap_left_disc.evaluate(None, linear_y), 0)
        self.assertEqual(extrap_right_disc.evaluate(None, linear_y), 3)

        # Fluxes
        extrap_flux_left = pybamm.BoundaryFlux(2 * var, "left")
        extrap_flux_right = pybamm.BoundaryFlux(1 - var, "right")
        extrap_flux_left_disc = disc.process_symbol(extrap_flux_left)
        extrap_flux_right_disc = disc.process_symbol(extrap_flux_right)

        # check constant extrapolates to constant
        self.assertEqual(extrap_flux_left_disc.evaluate(None, constant_y), 0)
        self.assertEqual(extrap_flux_right_disc.evaluate(None, constant_y), 0)

        # check linear variable extrapolates correctly
        self.assertEqual(extrap_flux_left_disc.evaluate(None, linear_y), 2)
        self.assertEqual(extrap_flux_right_disc.evaluate(None, linear_y), -1)

        # Microscale
        # create variable
        var = pybamm.Variable("var", domain="negative particle")
        surf_eqn = pybamm.surf(var)
        disc.set_variable_slices([var])
        surf_eqn_disc = disc.process_symbol(surf_eqn)

        # check constant extrapolates to constant
        constant_y = np.ones_like(micro_submesh[0].nodes[:, np.newaxis])
        self.assertEqual(surf_eqn_disc.evaluate(None, constant_y), 1)

        # check linear variable extrapolates correctly
        linear_y = micro_submesh[0].nodes
        y_surf = micro_submesh[0].nodes[-1] + micro_submesh[0].d_nodes[-1] / 2
        np.testing.assert_array_almost_equal(
            surf_eqn_disc.evaluate(None, linear_y), y_surf
        )

    def test_extrapolate_2d_models(self):
        # create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Microscale
        var = pybamm.Variable("var", domain="negative particle")
        extrap_right = pybamm.BoundaryValue(var, "right")
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        self.assertEqual(extrap_right_disc.domain, [])
        # domain for boundary values must now be explicitly set
        extrap_right.domain = ["negative electrode"]
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        self.assertEqual(extrap_right_disc.domain, ["negative electrode"])
        # evaluate
        y_macro = mesh["negative electrode"][0].nodes
        y_micro = mesh["negative particle"][0].nodes
        y = np.outer(y_macro, y_micro).reshape(-1, 1)
        # extrapolate to r=1 --> should evaluate to y_macro
        np.testing.assert_array_almost_equal(
            extrap_right_disc.evaluate(y=y)[:, 0], y_macro
        )

        var = pybamm.Variable("var", domain="positive particle")
        extrap_right = pybamm.BoundaryValue(var, "right")
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        self.assertEqual(extrap_right_disc.domain, [])
        # domain for boundary values must now be explicitly set
        extrap_right.domain = ["positive electrode"]
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        self.assertEqual(extrap_right_disc.domain, ["positive electrode"])

        # 2d macroscale
        mesh = get_1p1d_mesh_for_testing()
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        extrap_right = pybamm.BoundaryValue(var, "right")
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        self.assertEqual(extrap_right_disc.domain, [])
        # domain for boundary values must now be explicitly set
        extrap_right.domain = ["current collector"]
        disc.set_variable_slices([var])
        extrap_right_disc = disc.process_symbol(extrap_right)
        self.assertEqual(extrap_right_disc.domain, ["current collector"])

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
        y_test = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        for eqn in [
            var * pybamm.grad(var),
            var ** 2 * pybamm.grad(var),
            var * pybamm.grad(var) ** 2,
            var * (pybamm.grad(var) + 2),
            (pybamm.grad(var) + 2) * (-var),
            (pybamm.grad(var) + 2) * (2 * var),
            pybamm.grad(var) * pybamm.grad(var),
            (pybamm.grad(var) + 2) * pybamm.grad(var) ** 2,
            pybamm.div(pybamm.grad(var)),
            pybamm.div(pybamm.grad(var)) + 2,
            pybamm.div(pybamm.grad(var)) + var,
            pybamm.div(2 * pybamm.grad(var)),
            pybamm.div(2 * pybamm.grad(var)) + 3 * var,
            -2 * pybamm.div(var * pybamm.grad(var) + 2 * pybamm.grad(var)),
        ]:
            # Check that the equation can be evaluated in each case
            # Dirichlet
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # Neumann
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
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
        discretised_symbol = pybamm.StateVector(*disc.y_slices[var.id])
        bcs = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(3), "Dirichlet"),
        }

        # Test
        sym_ghost = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            var, discretised_symbol, bcs
        )
        combined_submesh = mesh.combine_submeshes(*whole_cell)
        y_test = np.linspace(0, 1, combined_submesh[0].npts)
        np.testing.assert_array_equal(
            sym_ghost.evaluate(y=y_test)[1:-1], discretised_symbol.evaluate(y=y_test)
        )
        self.assertEqual(
            (sym_ghost.evaluate(y=y_test)[0] + sym_ghost.evaluate(y=y_test)[1]) / 2, 0
        )
        self.assertEqual(
            (sym_ghost.evaluate(y=y_test)[-2] + sym_ghost.evaluate(y=y_test)[-1]) / 2, 3
        )

        # test errors
        bcs = {"left": (pybamm.Scalar(0), "x"), "right": (pybamm.Scalar(3), "Neumann")}
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            pybamm.FiniteVolume(mesh).add_ghost_nodes(var, discretised_symbol, bcs)
        bcs = {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(3), "x")}
        with self.assertRaisesRegex(ValueError, "boundary condition must be"):
            pybamm.FiniteVolume(mesh).add_ghost_nodes(var, discretised_symbol, bcs)

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
        bcs = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(3), "Dirichlet"),
        }

        # Test
        combined_submesh = mesh.combine_submeshes(*whole_cell)
        y_test = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])

        # both
        symbol_plus_ghost_both = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            var, discretised_symbol, bcs
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
        disc_c_s_n = pybamm.StateVector(*disc.y_slices[c_s_n.id])

        disc.set_variable_slices([c_s_p])
        disc_c_s_p = pybamm.StateVector(*disc.y_slices[c_s_p.id])
        bcs = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(3), "Dirichlet"),
        }
        c_s_n_plus_ghost = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            c_s_n, disc_c_s_n, bcs
        )
        c_s_p_plus_ghost = pybamm.FiniteVolume(mesh).add_ghost_nodes(
            c_s_p, disc_c_s_p, bcs
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
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[:, np.newaxis]),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh[0].nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        disc.bcs = boundary_conditions

        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[:, np.newaxis]),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh[0].nodes[:, np.newaxis]),
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
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[:, np.newaxis]),
        )

        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        y_linear = combined_submesh[0].nodes
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, y_linear),
            np.ones_like(combined_submesh[0].edges[:, np.newaxis]),
        )

        # div: test on linear r^2
        # div (grad r^2) = 6
        const = 6 * np.ones(combined_submesh[0].npts)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(6), "Dirichlet"),
                "right": (pybamm.Scalar(6), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, const), np.zeros((combined_submesh[0].npts, 1))
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
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        prim_pts = n_mesh[0].npts
        sec_pts = len(n_mesh)
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))

        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])

        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts + 1]))

        # div
        # div (grad r^2) = 6, N_left = N_right = 0
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        bc_var = disc.process_symbol(
            pybamm.SpatialVariable("x_n", domain="negative electrode")
        )
        boundary_conditions = {
            var.id: {"left": (bc_var, "Neumann"), "right": (bc_var, "Neumann")}
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        const = 6 * np.ones(sec_pts * prim_pts)
        div_eval = div_eqn_disc.evaluate(None, const)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_array_almost_equal(
            div_eval[:, :-1], np.zeros([sec_pts, prim_pts - 1])
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

        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[1:-1][:, np.newaxis]),
        )

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Linear y should have laplacian zero
        linear_y = combined_submesh[0].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[1:-1][:, np.newaxis]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh[0].nodes[:, np.newaxis]),
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
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Constant y should have gradient and laplacian zero
        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[:, np.newaxis]),
        )
        np.testing.assert_array_equal(
            div_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].nodes[:, np.newaxis]),
        )

        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Linear y should have gradient one and laplacian zero
        linear_y = combined_submesh[0].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[:, np.newaxis]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh[0].nodes[:, np.newaxis]),
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

        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[1:-1][:, np.newaxis]),
        )

        linear_y = combined_submesh[0].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[1:-1][:, np.newaxis]),
        )
        # div
        # div ( grad(r^2) ) == 6 , N_left = N_right = 0
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)

        linear_y = combined_submesh[0].nodes
        const = 6 * np.ones(combined_submesh[0].npts)

        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, const), np.zeros((combined_submesh[0].npts, 1))
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
            var.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
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
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])

        grad_eqn_disc = disc.process_symbol(grad_eqn)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh[0].edges[:, np.newaxis]),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh[0].nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(combined_submesh[0].edges[-1]), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh[0].edges[:, np.newaxis]),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh[0].nodes[:, np.newaxis]),
        )

    def test_definite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing(xpts=200, rpts=200)
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
        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ln + ls)
        linear_y = combined_submesh[0].nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), (ln + ls) ** 2 / 2
        )
        cos_y = np.cos(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y), np.sin(ln + ls), decimal=4
        )

        # domain not starting at zero
        var = pybamm.Variable("var", domain=["separator", "positive electrode"])
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        combined_submesh = mesh.combine_submeshes("separator", "positive electrode")
        constant_y = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ls + lp)
        linear_y = combined_submesh[0].nodes
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, linear_y)[0][0], (1 - (ln) ** 2) / 2
        )
        cos_y = np.cos(combined_submesh[0].nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y), np.sin(1) - np.sin(ln), decimal=4
        )

        # microscale variable
        var = pybamm.Variable("var", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", ["negative particle"])
        integral_eqn = pybamm.Integral(var, r)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        constant_y = np.ones_like(mesh["negative particle"][0].nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, constant_y), 2 * np.pi ** 2
        )
        linear_y = mesh["negative particle"][0].nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), 4 * np.pi ** 2 / 3, decimal=3
        )
        one_over_y = 1 / mesh["negative particle"][0].nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, one_over_y), 4 * np.pi ** 2
        )

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
        # Set boundary conditions (required for shape but don't matter)
        disc._bcs = {
            phi.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        int_grad_phi_disc = disc.process_symbol(int_grad_phi)
        left_boundary_value = pybamm.BoundaryValue(int_grad_phi, "left")
        left_boundary_value_disc = disc.process_symbol(left_boundary_value)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")

        # constant case
        phi_exact = np.ones((combined_submesh[0].npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)
        # linear case
        phi_exact = combined_submesh[0].nodes[:, np.newaxis]
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # sine case
        phi_exact = np.sin(combined_submesh[0].nodes[:, np.newaxis])
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # --------------------------------------------------------------------
        # region which doesn't start at zero
        phi = pybamm.Variable("phi", domain=["separator", "positive electrode"])
        i = pybamm.grad(phi)  # create test current (variable on edges)
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])
        int_grad_phi = pybamm.IndefiniteIntegral(i, x)
        disc.set_variable_slices([phi])  # i is not a fundamental variable
        disc._bcs = {
            phi.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        int_grad_phi_disc = disc.process_symbol(int_grad_phi)
        left_boundary_value = pybamm.BoundaryValue(int_grad_phi, "left")
        left_boundary_value_disc = disc.process_symbol(left_boundary_value)
        combined_submesh = mesh.combine_submeshes("separator", "positive electrode")

        # constant case
        phi_exact = np.ones((combined_submesh[0].npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # linear case
        phi_exact = (
            combined_submesh[0].nodes[:, np.newaxis] - combined_submesh[0].edges[0]
        )
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # sine case
        phi_exact = np.sin(
            combined_submesh[0].nodes[:, np.newaxis] - combined_submesh[0].edges[0]
        )
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # --------------------------------------------------------------------
        # micrsoscale case
        c = pybamm.Variable("c", domain=["negative particle"])
        N = pybamm.grad(c)  # create test current (variable on edges)
        r_n = pybamm.SpatialVariable("r_n", ["negative particle"])
        c_integral = pybamm.IndefiniteIntegral(N, r_n)
        disc.set_variable_slices([c])  # N is not a fundamental variable
        disc._bcs = {
            c.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

        c_integral_disc = disc.process_symbol(c_integral)
        left_boundary_value = pybamm.BoundaryValue(c_integral, "left")
        left_boundary_value_disc = disc.process_symbol(left_boundary_value)
        combined_submesh = mesh["negative particle"]

        # constant case
        c_exact = np.ones((combined_submesh[0].npts, 1))
        c_approx = c_integral_disc.evaluate(None, c_exact)
        c_approx += 1  # add constant of integration
        np.testing.assert_array_equal(c_exact, c_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=c_exact), 0)

        # linear case
        c_exact = combined_submesh[0].nodes[:, np.newaxis]
        c_approx = c_integral_disc.evaluate(None, c_exact)
        np.testing.assert_array_almost_equal(c_exact, c_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=c_exact), 0)

        # sine case
        c_exact = np.sin(combined_submesh[0].nodes[:, np.newaxis])
        c_approx = c_integral_disc.evaluate(None, c_exact)
        np.testing.assert_array_almost_equal(c_exact, c_approx, decimal=3)
        self.assertEqual(left_boundary_value_disc.evaluate(y=c_exact), 0)

    def test_indefinite_integral_on_nodes(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # input a phi, take grad, then integrate to recover phi approximation
        # (need to test this way as check evaluated on edges using if has grad
        # and no div)
        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])

        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        int_phi = pybamm.IndefiniteIntegral(phi, x)
        disc.set_variable_slices([phi])
        # Set boundary conditions (required for shape but don't matter)
        int_phi_disc = disc.process_symbol(int_phi)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")

        # constant case
        phi_exact = np.ones((combined_submesh[0].npts, 1))
        int_phi_exact = combined_submesh[0].edges
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact)[:, 0]
        np.testing.assert_array_equal(int_phi_exact, int_phi_approx)
        # linear case
        phi_exact = combined_submesh[0].nodes[:, np.newaxis]
        int_phi_exact = combined_submesh[0].edges[:, np.newaxis] ** 2 / 2
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(int_phi_exact, int_phi_approx)
        # cos case
        phi_exact = np.cos(combined_submesh[0].nodes[:, np.newaxis])
        int_phi_exact = np.sin(combined_submesh[0].edges[:, np.newaxis])
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(int_phi_exact, int_phi_approx, decimal=5)

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
            x1_disc.evaluate(), disc.mesh["negative electrode"][0].nodes[:, np.newaxis]
        )

        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x2_disc = disc.process_symbol(x2)
        self.assertIsInstance(x2_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x2_disc.evaluate(),
            disc.mesh.combine_submeshes("negative electrode", "separator")[0].nodes[
                :, np.newaxis
            ],
        )

        r = 3 * pybamm.SpatialVariable("r", ["negative particle"])
        r_disc = disc.process_symbol(r)
        self.assertIsInstance(r_disc.children[1], pybamm.Vector)
        np.testing.assert_array_equal(
            r_disc.evaluate(),
            3 * disc.mesh["negative particle"][0].nodes[:, np.newaxis],
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
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
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
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
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

    def test_jacobian(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y = pybamm.StateVector(slice(0, combined_submesh[0].npts))
        y_test = np.ones_like(combined_submesh[0].nodes[:, np.newaxis])

        # grad
        eqn = pybamm.grad(var)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Dirichlet"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        jacobian = eqn_jac.evaluate(y=y_test)
        grad_matrix = pybamm.FiniteVolume.gradient_matrix(disc, whole_cell).entries
        np.testing.assert_array_equal(jacobian.toarray()[1:-1], grad_matrix.toarray())
        np.testing.assert_array_equal(
            jacobian.toarray()[0, 0], grad_matrix.toarray()[0][0] * -2
        )
        np.testing.assert_array_equal(
            jacobian.toarray()[-1, -1], grad_matrix.toarray()[-1][-1] * -2
        )

        # grad with averaging
        eqn = var * pybamm.grad(var)
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

        # div(grad)
        flux = pybamm.grad(var)
        eqn = pybamm.div(flux)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

        # div(grad) with averaging
        flux = var * pybamm.grad(var)
        eqn = pybamm.div(flux)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

    def test_boundary_value_domain(self):
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

        disc.set_variable_slices([c_s_n, c_s_p])

        # surface values
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)

        # domain for boundary values must now be explicitly set
        c_s_n_surf_disc = disc.process_symbol(c_s_n_surf)
        c_s_p_surf_disc = disc.process_symbol(c_s_p_surf)
        self.assertEqual(c_s_n_surf_disc.domain, [])
        self.assertEqual(c_s_p_surf_disc.domain, [])
        c_s_n_surf.domain = ["negative electrode"]
        c_s_p_surf.domain = ["positive electrode"]
        c_s_n_surf_disc = disc.process_symbol(c_s_n_surf)
        c_s_p_surf_disc = disc.process_symbol(c_s_p_surf)
        self.assertEqual(c_s_n_surf_disc.domain, ["negative electrode"])
        self.assertEqual(c_s_p_surf_disc.domain, ["positive electrode"])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
