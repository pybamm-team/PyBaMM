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
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)
        n = mesh["negative electrode"].npts

        # node to edge
        c = pybamm.Vector(np.ones(n), domain=["negative electrode"])
        diffusivity_c_ari = fin_vol.node_to_edge(c, method="arithmetic")
        np.testing.assert_array_equal(diffusivity_c_ari.evaluate(), np.ones((n + 1, 1)))
        diffusivity_c_har = fin_vol.node_to_edge(c, method="harmonic")
        np.testing.assert_array_equal(diffusivity_c_har.evaluate(), np.ones((n + 1, 1)))

        # edge to node
        d = pybamm.StateVector(slice(0, n + 1), domain=["negative electrode"])
        y_test = np.ones(n + 1)
        diffusivity_d_ari = fin_vol.edge_to_node(d, method="arithmetic")
        np.testing.assert_array_equal(
            diffusivity_d_ari.evaluate(None, y_test), np.ones((n, 1))
        )
        diffusivity_d_har = fin_vol.edge_to_node(d, method="harmonic")
        np.testing.assert_array_equal(
            diffusivity_d_har.evaluate(None, y_test), np.ones((n, 1))
        )

        # bad shift key
        with self.assertRaisesRegex(ValueError, "shift key"):
            fin_vol.shift(c, "bad shift key", "arithmetic")

        with self.assertRaisesRegex(ValueError, "shift key"):
            fin_vol.shift(c, "bad shift key", "harmonic")

        # bad method
        with self.assertRaisesRegex(ValueError, "method"):
            fin_vol.shift(c, "shift key", "bad method")

    def test_concatenation(self):
        mesh = get_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        edges = [pybamm.Vector(mesh[dom].edges, domain=dom) for dom in whole_cell]
        # Concatenation of edges should get averaged to nodes first, using edge_to_node
        v_disc = fin_vol.concatenation(edges)
        np.testing.assert_array_equal(
            v_disc.evaluate()[:, 0], mesh.combine_submeshes(*whole_cell).nodes
        )

        # test for bad shape
        edges = [
            pybamm.Vector(np.ones(mesh[dom].npts + 2), domain=dom) for dom in whole_cell
        ]
        with self.assertRaisesRegex(pybamm.ShapeError, "child must have size n_nodes"):
            fin_vol.concatenation(edges)

    def test_discretise_diffusivity_times_spatial_operator(self):
        # Set up
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # Discretise some equations where averaging is needed
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y_test = np.ones_like(combined_submesh.nodes[:, np.newaxis])
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
            pybamm.laplacian(var),
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

    def test_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
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

        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:, np.newaxis]),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh.nodes
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
            np.ones_like(combined_submesh.edges[:, np.newaxis]),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )

    def test_grad_1plus1d(self):
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        a = pybamm.Variable(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        b = pybamm.Variable(
            "b",
            domain=["separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        c = pybamm.Variable(
            "c",
            domain=["positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        var = pybamm.Concatenation(a, b, c)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
                "right": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
            }
        }

        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(pybamm.grad(var))

        # Evaulate
        combined_submesh = mesh.combine_submeshes(*var.domain)
        linear_y = np.outer(np.linspace(0, 1, 15), combined_submesh.nodes).reshape(
            -1, 1
        )

        expected = np.outer(
            np.linspace(0, 1, 15), np.ones_like(combined_submesh.edges)
        ).reshape(-1, 1)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y), expected
        )

    def test_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        """
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        submesh = mesh["negative particle"]

        # grad
        # grad(r) == 1
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
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

        total_npts = (
            submesh.npts
            * mesh["negative electrode"].npts
            * mesh["current collector"].npts
        )
        total_npts_edges = (
            (submesh.npts + 1)
            * mesh["negative electrode"].npts
            * mesh["current collector"].npts
        )
        constant_y = np.ones((total_npts, 1))
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y), np.zeros((total_npts_edges, 1))
        )

        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        y_linear = np.tile(
            submesh.nodes,
            mesh["negative electrode"].npts * mesh["current collector"].npts,
        )
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, y_linear), np.ones((total_npts_edges, 1))
        )

        # div: test on linear r^2
        # div (grad r^2) = 6
        const = 6 * np.ones((total_npts, 1))
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
            div_eqn_disc.evaluate(None, const),
            np.zeros(
                (
                    submesh.npts
                    * mesh["negative electrode"].npts
                    * mesh["current collector"].npts,
                    1,
                )
            ),
        )

    def test_p2d_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        in the pseudo 2-dimensional case
        """

        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n_mesh = mesh["negative particle"]

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
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

        prim_pts = n_mesh.npts
        sec_pts = mesh["negative electrode"].npts
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
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[1:-1][:, np.newaxis]),
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
        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[1:-1][:, np.newaxis]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )

    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on c) on
        one side and Neumann boundary conditions (applied by div on N) on the other
        """
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
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
        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:, np.newaxis]),
        )
        np.testing.assert_array_equal(
            div_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
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
        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[:, np.newaxis]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )

    def test_spherical_grad_div_shapes_Neumann_bcs(self):
        """Test grad and div with Neumann boundary conditions (applied by div on N)"""

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes("negative particle")

        # grad
        var = pybamm.Variable("var", domain="negative particle")
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[1:-1][:, np.newaxis]),
        )

        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[1:-1][:, np.newaxis]),
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

        linear_y = combined_submesh.nodes
        const = 6 * np.ones(combined_submesh.npts)

        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, const), np.zeros((combined_submesh.npts, 1))
        )

    def test_p2d_spherical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions (applied by grad on var)
        in the pseudo 2-dimensional case
        """

        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n_mesh = mesh["negative particle"]

        mesh.add_ghost_meshes()
        disc.mesh.add_ghost_meshes()

        # test grad
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        grad_eqn = pybamm.grad(var)
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        prim_pts = n_mesh.npts
        sec_pts = mesh["negative electrode"].npts
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
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
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
        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(combined_submesh.edges[:, np.newaxis]),
        )

        # div: test on linear y (should have laplacian zero) so change bcs
        linear_y = combined_submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(combined_submesh.edges[-1]), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(combined_submesh.edges[:, np.newaxis]),
        )

        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(combined_submesh.nodes[:, np.newaxis]),
        )

    def test_definite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing(xpts=200, rpts=200)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges[-1]
        ls = mesh["separator"].edges[-1] - ln
        lp = 1 - (ln + ls)

        # macroscale variable
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ln + ls)
        linear_y = combined_submesh.nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), (ln + ls) ** 2 / 2
        )
        cos_y = np.cos(combined_submesh.nodes[:, np.newaxis])
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
        constant_y = np.ones_like(combined_submesh.nodes[:, np.newaxis])
        self.assertEqual(integral_eqn_disc.evaluate(None, constant_y), ls + lp)
        linear_y = combined_submesh.nodes
        self.assertAlmostEqual(
            integral_eqn_disc.evaluate(None, linear_y)[0][0], (1 - (ln) ** 2) / 2
        )
        cos_y = np.cos(combined_submesh.nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y), np.sin(1) - np.sin(ln), decimal=4
        )

        # microscale variable
        var = pybamm.Variable("var", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", ["negative particle"])
        integral_eqn = pybamm.Integral(var, r)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        constant_y = np.ones_like(mesh["negative particle"].nodes[:, np.newaxis])
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, constant_y), 4 * np.pi / 3, decimal=4
        )
        linear_y = mesh["negative particle"].nodes
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_y), np.pi, decimal=3
        )
        one_over_y_squared = 1 / mesh["negative particle"].nodes ** 2
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, one_over_y_squared), 4 * np.pi
        )

        # test failure for secondary dimension column form
        finite_volume = pybamm.FiniteVolume()
        finite_volume.build(mesh)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Integral in secondary vector only implemented in 'row' form",
        ):
            finite_volume.definite_integral_matrix(var, "column", "secondary")

    def test_integral_secondary_domain(self):
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges[-1]
        ls = mesh["separator"].edges[-1] - ln
        lp = 1 - (ln + ls)

        var = pybamm.Variable(
            "var",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )
        x = pybamm.SpatialVariable("x", "positive electrode")
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        submesh = mesh["positive particle"]
        constant_y = np.ones(
            (
                submesh.npts
                * mesh["positive electrode"].npts
                * mesh["current collector"].npts,
                1,
            )
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, constant_y),
            lp * np.ones((submesh.npts * mesh["current collector"].npts, 1)),
        )
        linear_in_x = np.tile(
            np.repeat(mesh["positive electrode"].nodes, submesh.npts),
            mesh["current collector"].npts,
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_in_x),
            (1 - (ln + ls) ** 2)
            / 2
            * np.ones((submesh.npts * mesh["current collector"].npts, 1)),
        )
        linear_in_r = np.tile(
            submesh.nodes,
            mesh["positive electrode"].npts * mesh["current collector"].npts,
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, linear_in_r).flatten(),
            lp * np.tile(submesh.nodes, mesh["current collector"].npts),
        )
        cos_y = np.cos(linear_in_x)
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, cos_y),
            (np.sin(1) - np.sin(ln + ls))
            * np.ones((submesh.npts * mesh["current collector"].npts, 1)),
            decimal=4,
        )

    def test_integral_primary_then_secondary_same_result(self):
        # Test that integrating in r then in x gives the same result as integrating in
        # x then in r
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable(
            "var",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )
        x = pybamm.SpatialVariable("x", "positive electrode")
        r = pybamm.SpatialVariable("r", "positive particle")
        integral_eqn_x_then_r = pybamm.Integral(pybamm.Integral(var, x), r)
        integral_eqn_r_then_x = pybamm.Integral(pybamm.Integral(var, r), x)

        # discretise
        disc.set_variable_slices([var])
        integral_eqn_x_then_r_disc = disc.process_symbol(integral_eqn_x_then_r)
        integral_eqn_r_then_x_disc = disc.process_symbol(integral_eqn_r_then_x)

        # test
        submesh = mesh["positive particle"]
        cos_y = np.cos(
            np.tile(
                submesh.nodes,
                mesh["positive electrode"].npts * mesh["current collector"].npts,
            )
        )
        np.testing.assert_array_almost_equal(
            integral_eqn_x_then_r_disc.evaluate(None, cos_y),
            integral_eqn_r_then_x_disc.evaluate(None, cos_y),
            decimal=4,
        )

    def test_integral_secondary_domain_on_edges_in_primary_domain(self):
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges[-1]
        ls = mesh["separator"].edges[-1] - ln
        lp = 1 - (ln + ls)

        r_edge = pybamm.SpatialVariableEdge(
            "r_p",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )

        x = pybamm.SpatialVariable("x", "positive electrode")
        integral_eqn = pybamm.Integral(r_edge, x)
        integral_eqn_disc = disc.process_symbol(integral_eqn)

        submesh = mesh["positive particle"]
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate().flatten(),
            lp
            * np.tile(
                np.linspace(0, 1, submesh.npts + 1), mesh["current collector"].npts
            ),
        )

    def test_definite_integral_vector(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="negative electrode")
        disc.set_variable_slices([var])

        # row (default)
        vec = pybamm.DefiniteIntegralVector(var)
        vec_disc = disc.process_symbol(vec)
        self.assertEqual(vec_disc.shape[0], 1)
        self.assertEqual(vec_disc.shape[1], mesh["negative electrode"].npts)

        # column
        vec = pybamm.DefiniteIntegralVector(var, vector_type="column")
        vec_disc = disc.process_symbol(vec)
        self.assertEqual(vec_disc.shape[0], mesh["negative electrode"].npts)
        self.assertEqual(vec_disc.shape[1], 1)

    def test_indefinite_integral(self):

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
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
        phi_exact = np.ones((combined_submesh.npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)
        # linear case
        phi_exact = combined_submesh.nodes[:, np.newaxis]
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # sine case
        phi_exact = np.sin(combined_submesh.nodes[:, np.newaxis])
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
        phi_exact = np.ones((combined_submesh.npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=phi_exact), 0)

        # linear case
        phi_exact = combined_submesh.nodes[:, np.newaxis] - combined_submesh.edges[0]
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=phi_exact), 0
        )

        # sine case
        phi_exact = np.sin(
            combined_submesh.nodes[:, np.newaxis] - combined_submesh.edges[0]
        )
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(phi_exact, phi_approx)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=phi_exact), 0
        )

        # --------------------------------------------------------------------
        # indefinite integral of a spatial variable
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x_edge = pybamm.SpatialVariableEdge("x", ["negative electrode", "separator"])
        int_x = pybamm.IndefiniteIntegral(x, x)
        int_x_edge = pybamm.IndefiniteIntegral(x_edge, x)

        x_disc = disc.process_symbol(x)
        x_edge_disc = disc.process_symbol(x_edge)
        int_x_disc = disc.process_symbol(int_x)
        int_x_edge_disc = disc.process_symbol(int_x_edge)

        np.testing.assert_almost_equal(
            int_x_disc.evaluate(), x_edge_disc.evaluate() ** 2 / 2
        )
        np.testing.assert_almost_equal(
            int_x_edge_disc.evaluate(), x_disc.evaluate() ** 2 / 2, decimal=4
        )

        # --------------------------------------------------------------------
        # micrsoscale case
        c = pybamm.Variable("c", domain=["negative particle"])
        N = pybamm.grad(c)  # create test flux (variable on edges)
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
        c_exact = np.ones((combined_submesh.npts, 1))
        c_approx = c_integral_disc.evaluate(None, c_exact)
        c_approx += 1  # add constant of integration
        np.testing.assert_array_equal(c_exact, c_approx)
        self.assertEqual(left_boundary_value_disc.evaluate(y=c_exact), 0)

        # linear case
        c_exact = combined_submesh.nodes[:, np.newaxis]
        c_approx = c_integral_disc.evaluate(None, c_exact)
        np.testing.assert_array_almost_equal(c_exact, c_approx)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=c_exact), 0
        )

        # sine case
        c_exact = np.sin(combined_submesh.nodes[:, np.newaxis])
        c_approx = c_integral_disc.evaluate(None, c_exact)
        np.testing.assert_array_almost_equal(c_exact, c_approx, decimal=3)
        np.testing.assert_array_almost_equal(
            left_boundary_value_disc.evaluate(y=c_exact), 0
        )

    def test_backward_indefinite_integral(self):

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # --------------------------------------------------------------------
        # region which doesn't start at zero
        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        i = pybamm.grad(phi)  # create test current (variable on edges)
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        int_grad_phi = pybamm.BackwardIndefiniteIntegral(i, x)
        disc.set_variable_slices([phi])  # i is not a fundamental variable
        disc._bcs = {
            phi.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        int_grad_phi_disc = disc.process_symbol(int_grad_phi)
        right_boundary_value = pybamm.BoundaryValue(int_grad_phi, "right")
        right_boundary_value_disc = disc.process_symbol(right_boundary_value)
        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")

        # Test that the backward_integral(grad(phi)) = -phi
        # constant case
        phi_exact = np.ones((combined_submesh.npts, 1))
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact)
        phi_approx += 1  # add constant of integration
        np.testing.assert_array_equal(phi_exact, phi_approx)
        self.assertEqual(right_boundary_value_disc.evaluate(y=phi_exact), 0)

        # linear case
        phi_exact = combined_submesh.nodes - combined_submesh.edges[-1]
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(phi_exact, -phi_approx)
        np.testing.assert_array_almost_equal(
            right_boundary_value_disc.evaluate(y=phi_exact), 0
        )

        # sine case
        phi_exact = np.sin(combined_submesh.nodes - combined_submesh.edges[-1])
        phi_approx = int_grad_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(phi_exact, -phi_approx)
        np.testing.assert_array_almost_equal(
            right_boundary_value_disc.evaluate(y=phi_exact), 0
        )

    def test_indefinite_integral_of_broadcasted_to_cell_edges(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # make a variable 'phi' and a vector 'i' which is broadcast onto edges
        # the integral of this should then be put onto the nodes
        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        i = pybamm.PrimaryBroadcastToEdges(1, phi.domain)
        x = pybamm.SpatialVariable("x", phi.domain)
        disc.set_variable_slices([phi])
        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        x_end = combined_submesh.edges[-1]

        # take indefinite integral
        int_phi = pybamm.IndefiniteIntegral(i * phi, x)
        # take integral again
        int_int_phi = pybamm.Integral(int_phi, x)
        int_int_phi_disc = disc.process_symbol(int_int_phi)

        # constant case
        phi_exact = np.ones_like(combined_submesh.nodes)
        phi_approx = int_int_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_equal(x_end ** 2 / 2, phi_approx)

        # linear case
        phi_exact = combined_submesh.nodes[:, np.newaxis]
        phi_approx = int_int_phi_disc.evaluate(None, phi_exact)
        np.testing.assert_array_almost_equal(x_end ** 3 / 6, phi_approx, decimal=4)

    def test_indefinite_integral_on_nodes(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])

        int_phi = pybamm.IndefiniteIntegral(phi, x)
        disc.set_variable_slices([phi])
        int_phi_disc = disc.process_symbol(int_phi)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")

        # constant case
        phi_exact = np.ones((combined_submesh.npts, 1))
        int_phi_exact = combined_submesh.edges
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_equal(int_phi_exact, int_phi_approx)
        # linear case
        phi_exact = combined_submesh.nodes
        int_phi_exact = combined_submesh.edges ** 2 / 2
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(int_phi_exact, int_phi_approx)
        # cos case
        phi_exact = np.cos(combined_submesh.nodes)
        int_phi_exact = np.sin(combined_submesh.edges)
        int_phi_approx = int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(int_phi_exact, int_phi_approx, decimal=5)

        # microscale case should fail
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        c = pybamm.Variable("c", domain=["negative particle"])
        r = pybamm.SpatialVariable("r", ["negative particle"])

        int_c = pybamm.IndefiniteIntegral(c, r)
        disc.set_variable_slices([c])
        with self.assertRaisesRegex(
            NotImplementedError,
            "Indefinite integral on a spherical polar domain is not implemented",
        ):
            disc.process_symbol(int_c)

    def test_backward_indefinite_integral_on_nodes(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        phi = pybamm.Variable("phi", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", ["negative electrode", "separator"])

        back_int_phi = pybamm.BackwardIndefiniteIntegral(phi, x)
        disc.set_variable_slices([phi])
        back_int_phi_disc = disc.process_symbol(back_int_phi)

        combined_submesh = mesh.combine_submeshes("negative electrode", "separator")
        edges = combined_submesh.edges

        # constant case
        phi_exact = np.ones((combined_submesh.npts, 1))
        back_int_phi_exact = edges[-1] - edges
        back_int_phi_approx = back_int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(back_int_phi_exact, back_int_phi_approx)
        # linear case
        phi_exact = combined_submesh.nodes
        back_int_phi_exact = edges[-1] ** 2 / 2 - edges ** 2 / 2
        back_int_phi_approx = back_int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(back_int_phi_exact, back_int_phi_approx)
        # cos case
        phi_exact = np.cos(combined_submesh.nodes)
        back_int_phi_exact = np.sin(edges[-1]) - np.sin(edges)
        back_int_phi_approx = back_int_phi_disc.evaluate(None, phi_exact).flatten()
        np.testing.assert_array_almost_equal(
            back_int_phi_exact, back_int_phi_approx, decimal=5
        )

    def test_forward_plus_backward_integral(self):
        # Test that forward integral + backward integral = integral
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # On nodes
        phi = pybamm.Variable("phi", domain=["separator", "positive electrode"])
        x = pybamm.SpatialVariable("x", ["separator", "positive electrode"])

        disc.set_variable_slices([phi])

        full_int_phi = pybamm.PrimaryBroadcastToEdges(
            pybamm.Integral(phi, x), ["separator", "positive electrode"]
        )
        full_int_phi_disc = disc.process_symbol(full_int_phi)
        int_plus_back_int_phi = pybamm.IndefiniteIntegral(
            phi, x
        ) + pybamm.BackwardIndefiniteIntegral(phi, x)
        int_plus_back_int_phi_disc = disc.process_symbol(int_plus_back_int_phi)

        combined_submesh = mesh.combine_submeshes("separator", "positive electrode")

        # test
        for phi_exact in [
            np.ones((combined_submesh.npts, 1)),
            combined_submesh.nodes,
            np.cos(combined_submesh.nodes),
        ]:
            np.testing.assert_array_almost_equal(
                full_int_phi_disc.evaluate(y=phi_exact).flatten(),
                int_plus_back_int_phi_disc.evaluate(y=phi_exact).flatten(),
            )

    def test_discretise_spatial_variable(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # space
        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x1_disc = disc.process_symbol(x1)
        self.assertIsInstance(x1_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x1_disc.evaluate(), disc.mesh["negative electrode"].nodes[:, np.newaxis]
        )

        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x2_disc = disc.process_symbol(x2)
        self.assertIsInstance(x2_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x2_disc.evaluate(),
            disc.mesh.combine_submeshes("negative electrode", "separator").nodes[
                :, np.newaxis
            ],
        )

        r = 3 * pybamm.SpatialVariable("r", ["negative particle"])
        r_disc = disc.process_symbol(r)
        self.assertIsInstance(r_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            r_disc.evaluate(), 3 * disc.mesh["negative particle"].nodes[:, np.newaxis],
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
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)
        disc.process_model(model)

        # mass matrix
        mass = np.eye(combined_submesh.npts)
        np.testing.assert_array_equal(mass, model.mass_matrix.entries.toarray())

    def test_p2d_mass_matrix_shape(self):
        """
        Test mass matrix shape in the pseudo 2-dimensional case
        """
        c = pybamm.Variable(
            "c",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.variables = {"c": c, "N": N}
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        prim_pts = mesh["negative particle"].npts
        sec_pts = mesh["negative electrode"].npts
        mass_local = eye(prim_pts)
        mass = kron(eye(sec_pts), mass_local)
        np.testing.assert_array_equal(
            mass.toarray(), model.mass_matrix.entries.toarray()
        )

    def test_jacobian(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y = pybamm.StateVector(slice(0, combined_submesh.npts))
        y_test = np.ones_like(combined_submesh.nodes[:, np.newaxis])

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
        spatial_method = pybamm.FiniteVolume()
        spatial_method.build(mesh)
        grad_matrix = spatial_method.gradient_matrix(whole_cell, {}).entries
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
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

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

    def test_delta_function(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var")
        delta_fn_left = pybamm.DeltaFunction(var, "left", "negative electrode")
        delta_fn_right = pybamm.DeltaFunction(var, "right", "negative electrode")
        disc.set_variable_slices([var])
        delta_fn_left_disc = disc.process_symbol(delta_fn_left)
        delta_fn_right_disc = disc.process_symbol(delta_fn_right)

        # Basic shape and type tests
        y = np.ones_like(mesh["negative electrode"].nodes[:, np.newaxis])
        # Left
        self.assertEqual(delta_fn_left_disc.domain, delta_fn_left.domain)
        self.assertEqual(
            delta_fn_left_disc.auxiliary_domains, delta_fn_left.auxiliary_domains
        )
        self.assertIsInstance(delta_fn_left_disc, pybamm.Multiplication)
        self.assertIsInstance(delta_fn_left_disc.left, pybamm.Matrix)
        np.testing.assert_array_equal(delta_fn_left_disc.left.evaluate()[:, 1:], 0)
        self.assertEqual(delta_fn_left_disc.shape, y.shape)
        # Right
        self.assertEqual(delta_fn_right_disc.domain, delta_fn_right.domain)
        self.assertEqual(
            delta_fn_right_disc.auxiliary_domains, delta_fn_right.auxiliary_domains
        )
        self.assertIsInstance(delta_fn_right_disc, pybamm.Multiplication)
        self.assertIsInstance(delta_fn_right_disc.left, pybamm.Matrix)
        np.testing.assert_array_equal(delta_fn_right_disc.left.evaluate()[:, :-1], 0)
        self.assertEqual(delta_fn_right_disc.shape, y.shape)

        # Value tests
        # Delta function should integrate to the same thing as variable
        var_disc = disc.process_symbol(var)
        x = pybamm.standard_spatial_vars.x_n
        delta_fn_int_disc = disc.process_symbol(pybamm.Integral(delta_fn_left, x))
        np.testing.assert_array_equal(
            var_disc.evaluate(y=y) * mesh["negative electrode"].edges[-1],
            np.sum(delta_fn_int_disc.evaluate(y=y)),
        )

    def test_heaviside(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="negative electrode")
        heav = var > 1

        disc.set_variable_slices([var])
        # process_binary_operators should work with heaviside
        disc_heav = disc.process_symbol(heav * var)
        nodes = mesh["negative electrode"].nodes
        self.assertEqual(disc_heav.size, nodes.size)
        np.testing.assert_array_equal(disc_heav.evaluate(y=2 * np.ones_like(nodes)), 2)
        np.testing.assert_array_equal(disc_heav.evaluate(y=-2 * np.ones_like(nodes)), 0)

    def test_upwind_downwind(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n = mesh["negative electrode"].npts
        var = pybamm.StateVector(slice(0, n), domain="negative electrode")
        upwind = pybamm.upwind(var)
        downwind = pybamm.downwind(var)

        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(5), "Dirichlet"),
                "right": (pybamm.Scalar(3), "Dirichlet"),
            }
        }

        disc_upwind = disc.process_symbol(upwind)
        disc_downwind = disc.process_symbol(downwind)

        nodes = mesh["negative electrode"].nodes
        n = mesh["negative electrode"].npts
        self.assertEqual(disc_upwind.size, nodes.size + 1)
        self.assertEqual(disc_downwind.size, nodes.size + 1)

        y_test = 2 * np.ones_like(nodes)
        np.testing.assert_array_equal(
            disc_upwind.evaluate(y=y_test),
            np.concatenate([np.array([5, 0.5]), 2 * np.ones(n - 1)])[:, np.newaxis],
        )
        np.testing.assert_array_equal(
            disc_downwind.evaluate(y=y_test),
            np.concatenate([2 * np.ones(n - 1), np.array([1.5, 3])])[:, np.newaxis],
        )

        # Remove boundary conditions and check error is raised
        disc.bcs = {}
        with self.assertRaisesRegex(pybamm.ModelError, "Boundary conditions"):
            disc.process_symbol(upwind)

        # Set wrong boundary conditions and check error is raised
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(5), "Neumann"),
                "right": (pybamm.Scalar(3), "Neumann"),
            }
        }
        with self.assertRaisesRegex(pybamm.ModelError, "Dirichlet boundary conditions"):
            disc.process_symbol(upwind)
        with self.assertRaisesRegex(pybamm.ModelError, "Dirichlet boundary conditions"):
            disc.process_symbol(downwind)

    def test_grad_div_with_bcs_on_tab(self):
        # 2d macroscale
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        y_test = np.ones(mesh["current collector"].npts)

        # var
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        # grad
        grad_eqn = pybamm.grad(var)
        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)

        # bcs (on each tab)
        boundary_conditions = {
            var.id: {
                "negative tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

        # bcs (one pos, one not tab)
        boundary_conditions = {
            var.id: {
                "no tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

        # bcs (one neg, one not tab)
        boundary_conditions = {
            var.id: {
                "negative tab": (pybamm.Scalar(1), "Neumann"),
                "no tab": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

    def test_neg_pos_bcs(self):
        # 2d macroscale
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # var
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        # grad
        grad_eqn = pybamm.grad(var)

        # bcs (on each tab)
        boundary_conditions = {
            var.id: {
                "negative tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
                "no tab": (pybamm.Scalar(8), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        # check after disc that negative tab goes to left and positive tab goes
        # to right
        disc.process_symbol(grad_eqn)
        self.assertEqual(disc.bcs[var.id]["left"][0].id, pybamm.Scalar(1).id)
        self.assertEqual(disc.bcs[var.id]["left"][1], "Dirichlet")
        self.assertEqual(disc.bcs[var.id]["right"][0].id, pybamm.Scalar(0).id)
        self.assertEqual(disc.bcs[var.id]["right"][1], "Neumann")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
