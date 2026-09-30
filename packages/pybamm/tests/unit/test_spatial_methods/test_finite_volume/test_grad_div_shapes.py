#
# Test for the gradient and divergence in Finite Volumes
#

import numpy as np
import pytest

import pybamm
from tests import (
    assert_constant_matrix_factors,
    assert_symbolic_mesh_matches_numeric,
    get_1p1d_mesh_for_testing,
    get_cylindrical_mesh_for_testing,
    get_cylindrical_mesh_for_testing_symbolic,
    get_mesh_for_testing,
    get_mesh_for_testing_symbolic,
    get_mesh_for_testing_symbolic_concatenation,
    get_p2d_mesh_for_testing,
    get_spherical_mesh_for_testing_symbolic,
    get_symbolic_length_discretisation_for_testing,
)


class TestFiniteVolumeGradDiv:
    def test_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions in Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient of constant is zero
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )

        # Test operations on linear x
        linear_y = submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_cylindrical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions in cylindrical polar
        coordinates
        """
        # Create discretisation
        mesh = get_cylindrical_mesh_for_testing()
        spatial_methods = {"current collector": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh["current collector"]
        npts = submesh.npts
        npts_edges = submesh.npts + 1

        # Test gradient of a constant is zero
        # grad(1) = 0
        constant_y = np.ones((npts, 1))
        var = pybamm.Variable(
            "var",
            domain=["current collector"],
        )
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y), np.zeros((npts_edges, 1))
        )

        # Test operations on linear and quadratic in r
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(submesh.edges[0]), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(r) == 1
        y_linear = submesh.nodes
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, y_linear),
            np.ones((npts_edges, 1)),
            rtol=1e-7,
            atol=1e-6,
        )
        # div(grad r^2) = 4
        y_squared = submesh.nodes**2
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, y_squared)
        np.testing.assert_allclose(
            div_eval[1:-1], 4 * np.ones((npts - 2, 1)), rtol=1e-7, atol=1e-6
        )

    def test_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions in spherical polar
        coordinates
        """
        # Create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh["negative particle"]
        npts = submesh.npts
        sec_npts = mesh["negative electrode"].npts * mesh["current collector"].npts
        total_npts = npts * sec_npts
        total_npts_edges = (npts + 1) * sec_npts

        # Test gradient
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        grad_eqn = pybamm.grad(var)
        # grad(1) = 0
        constant_y = np.ones((total_npts, 1))
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y), np.zeros((total_npts_edges, 1))
        )
        # grad(r) == 1
        y_linear = np.tile(submesh.nodes, sec_npts)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(submesh.edges[0]), "Dirichlet"),
                "right": (pybamm.Scalar(submesh.edges[-1]), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, y_linear),
            np.ones((total_npts_edges, 1)),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test divergence of gradient
        # div(grad r^2) = 6
        y_squared = np.tile(submesh.nodes**2, sec_npts)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(submesh.nodes[0]), "Dirichlet"),
                "right": (pybamm.Scalar(submesh.nodes[-1]), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, y_squared)
        div_eval = np.reshape(div_eval, [sec_npts, npts])
        np.testing.assert_allclose(
            div_eval[:, :-1], 6 * np.ones([sec_npts, npts - 1]), rtol=1e-7, atol=1e-6
        )

    def test_p2d_spherical_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions in the pseudo
        2-dimensional case
        """
        # Create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        prim_pts = mesh["negative particle"].npts
        sec_pts = mesh["negative electrode"].npts

        # Test gradient of a constant is zero
        # grad(1) = 0
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])
        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts + 1]))

        # Test divergence of gradient
        # div(grad r^2) = 6
        y_squared = np.tile(mesh["negative particle"].nodes ** 2, sec_pts)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, y_squared)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_allclose(
            div_eval[:, :-1], 6 * np.ones([sec_pts, prim_pts - 1]), rtol=1e-7, atol=1e-6
        )

    def test_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions in Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient of constant is zero
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )

        # Test operations on linear x
        linear_y = submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs(self):
        """
        Test grad and div with a Dirichlet boundary condition on one side and
        a Neumann boundary conditions on the other side in Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient and divergence of a constant
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        # grad(1) = 0
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )
        # div(grad(1)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test gradient and divergence of linear x
        linear_y = submesh.nodes
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_cylindrical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions in cylindrical polar
        coordinates
        """
        # Create discretisation
        mesh = get_cylindrical_mesh_for_testing()
        spatial_methods = {"current collector": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh["current collector"]
        npts = submesh.npts
        npts_edges = submesh.npts + 1

        # Test gradient
        var = pybamm.Variable("var", domain="current collector")
        grad_eqn = pybamm.grad(var)
        # grad(1) = 0
        constant_y = np.ones((npts, 1))
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y), np.zeros((npts_edges, 1))
        )
        # grad(r) = 1
        y_linear = submesh.nodes
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, y_linear),
            np.ones((npts_edges, 1)),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test divergence
        # div(grad(r^2)) = 4 , N_left = 2*r_inner, N_right = 2
        y_squared = submesh.nodes**2
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(2 * submesh.edges[0]), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, y_squared),
            4 * np.ones((npts, 1)),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_cylindrical_grad_div_shapes_Neumann_bcs_symbolic(self):
        mesh = get_cylindrical_mesh_for_testing_symbolic()
        spatial_methods = {"cylindrical domain": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Test gradient
        var = pybamm.Variable("var", domain="cylindrical domain")
        disc.set_variable_slices([var])
        grad_eqn = pybamm.grad(var)
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        constant_y = np.ones_like(mesh["cylindrical domain"].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros((14, 1)),
        )

        # Test divergence
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(pybamm.div(grad_eqn))
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, constant_y),
            np.zeros((15, 1)),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test divergence of gradient
        # div(grad(r^2)) = 4, N_left = 2*r_inner, N_right = 2
        submesh = mesh["cylindrical domain"]
        y_squared = (submesh.nodes * submesh.length) ** 2
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(4), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, y_squared),
            4 * np.ones((15, 1)),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_spherical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions spherical polar
        coordinates
        """
        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh["negative particle"]

        # Test gradient
        var = pybamm.Variable("var", domain="negative particle")
        grad_eqn = pybamm.grad(var)
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )
        # grad(r) == 1
        linear_y = submesh.nodes
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test divergence of gradient
        # div(grad(r^2)) = 6 , N_left = 2*0 = 0, N_right = 2*0.5=1
        quadratic_y = submesh.nodes**2
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, quadratic_y),
            6 * np.ones((submesh.npts, 1)),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_spherical_grad_div_shapes_Neumann_bcs_symbolic(self):
        mesh = get_spherical_mesh_for_testing_symbolic()
        spatial_methods = {"spherical domain": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Test gradient
        var = pybamm.Variable("var", domain="spherical domain")
        disc.set_variable_slices([var])
        grad_eqn = pybamm.grad(var)
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        constant_y = np.ones_like(mesh["spherical domain"].nodes[:, np.newaxis])
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros((14, 1)),
        )

        # Test divergence
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        div_eqn_disc = disc.process_symbol(pybamm.div(grad_eqn))
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(
                None, np.ones_like(mesh["spherical domain"].nodes[:, np.newaxis])
            ),
            np.zeros_like(mesh["spherical domain"].nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test divergence of gradient
        # div(grad(r^2)) = 6, N_left = 0, N_right = 2
        submesh = mesh["spherical domain"]
        quadratic_y = (submesh.nodes * submesh.length) ** 2
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(4), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, quadratic_y),
            6 * np.ones_like(mesh["spherical domain"].nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_p2d_spherical_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions in the pseudo
        2-dimensional case
        """
        # Create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        prim_pts = mesh["negative particle"].npts
        sec_pts = mesh["negative electrode"].npts

        # Test gradient of a constant is zero
        # grad(1) = 0
        constant_y = np.kron(np.ones(sec_pts), np.ones(prim_pts))
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eval = grad_eqn_disc.evaluate(None, constant_y)
        grad_eval = np.reshape(grad_eval, [sec_pts, prim_pts + 1])
        np.testing.assert_array_equal(grad_eval, np.zeros([sec_pts, prim_pts + 1]))

        # Test divergence of gradient
        # div(grad r^2) = 6, N_left = 0, N_right = 2
        submesh = mesh["negative particle"]
        y_squared = np.tile(submesh.nodes**2, sec_pts)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(2 * submesh.edges[-1]), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, y_squared)
        div_eval = np.reshape(div_eval, [sec_pts, prim_pts])
        np.testing.assert_allclose(
            div_eval, 6 * np.ones([sec_pts, prim_pts]), rtol=1e-7, atol=1e-6
        )

    def test_grad_div_shapes_mixed_domain(self):
        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[("negative electrode", "separator")]

        # Test gradient of constant
        # grad(1) = 0
        constant_y = np.ones_like(submesh.nodes[:, np.newaxis])
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(submesh.edges[:, np.newaxis]),
        )

        # Test operations on linear x
        linear_y = submesh.nodes
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(submesh.edges[-1]), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(submesh.edges[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
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
        var = pybamm.concatenation(a, b, c)
        boundary_conditions = {
            var: {
                "left": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
                "right": (pybamm.Vector(np.linspace(0, 1, 15)), "Neumann"),
            }
        }

        # Discretise
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(pybamm.grad(var))

        # Evaulate
        submesh = mesh[var.domain]
        linear_y = np.outer(np.linspace(0, 1, 15), submesh.nodes).reshape(-1, 1)
        expected = np.outer(np.linspace(0, 1, 15), np.ones_like(submesh.edges)).reshape(
            -1, 1
        )
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y), expected, rtol=1e-7, atol=1e-6
        )

    def test_grad_div_shapes_symbolic_mesh(self):
        mesh = get_mesh_for_testing_symbolic()
        spatial_methods = {"domain": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="domain")
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(grad_eqn)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Evaluate grad
        dom = ("domain_left ghost cell", "domain", "domain_right ghost cell")
        linear_y = mesh[dom].nodes * mesh[dom].length + mesh[dom].min
        expected = np.ones((16, 1))
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y), expected, rtol=1e-7, atol=1e-6
        )

        # Evaluate div
        div_eqn = pybamm.div(pybamm.grad(var))
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, linear_y)
        div_eval = np.reshape(div_eval, [15, 1])
        np.testing.assert_allclose(div_eval, np.zeros([15, 1]), rtol=1e-7, atol=1e-6)

    def test_grad_div_shapes_symbolic_mesh_concatenation(self):
        mesh = get_mesh_for_testing_symbolic_concatenation()
        spatial_methods = {
            "domain 1": pybamm.FiniteVolume(),
            "domain 2": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var_1 = pybamm.Variable("var", domain="domain 1")
        var_2 = pybamm.Variable("var", domain="domain 2")
        var = pybamm.concatenation(var_1, var_2)
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(grad_eqn)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # Evaluate grad
        dom = (
            "domain 1_left ghost cell",
            "domain 1",
            "domain 2",
            "domain 2_right ghost cell",
        )
        submeshes = [mesh[domain_] for domain_ in dom]
        nodes_list = []
        for submesh_ in submeshes:
            nodes_ = submesh_.nodes
            if hasattr(submesh_, "length"):
                nodes_ = nodes_ * submesh_.length + submesh_.min
            nodes_list.append(nodes_)
        linear_y = np.concatenate(nodes_list)
        expected = np.ones((31, 1))
        np.testing.assert_allclose(
            grad_eqn_disc.evaluate(None, linear_y), expected, rtol=1e-7, atol=1e-6
        )

        # Evaluate div
        div_eqn = pybamm.div(pybamm.grad(var))
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eval = div_eqn_disc.evaluate(None, linear_y)
        div_eval = np.reshape(div_eval, [30, 1])
        np.testing.assert_allclose(div_eval, np.zeros([30, 1]), rtol=1e-7, atol=1e-6)

    @pytest.mark.parametrize(
        "coord_sys", ["cartesian", "cylindrical polar", "spherical polar"]
    )
    @pytest.mark.parametrize("right_bc", ["Neumann", "Dirichlet"])
    def test_symbolic_length_mesh_with_secondary_domain(self, coord_sys, right_bc):
        auxiliary_domains = {"secondary": "electrode"}
        c = pybamm.Variable("c", "particle", auxiliary_domains=auxiliary_domains)
        r = pybamm.SpatialVariable(
            "r", ["particle"], auxiliary_domains=auxiliary_domains, coord_sys=coord_sys
        )
        r_edge = pybamm.SpatialVariableEdge(
            "r", ["particle"], auxiliary_domains=auxiliary_domains, coord_sys=coord_sys
        )
        expressions = [
            pybamm.grad(c),
            pybamm.div(pybamm.grad(c)),
            # a node-valued factor of a gradient takes the harmonic mean
            pybamm.div((1 + c**2) * pybamm.grad(c)),
            r,
            r_edge,
        ]

        def discretise(radius):
            disc = get_symbolic_length_discretisation_for_testing(
                radius, coord_sys=coord_sys
            )
            disc.set_variable_slices([c])
            disc.bcs = {
                c: {
                    "left": (pybamm.Scalar(1), "Neumann"),
                    "right": (pybamm.Scalar(2), right_bc),
                }
            }
            discretised = [disc.process_symbol(expr) for expr in expressions]
            method = disc.spatial_methods["particle"]
            discretised.append(method.edge_to_node(discretised[0], "harmonic"))
            return discretised

        y = 1 + np.linspace(0, 1, 18)[:, np.newaxis] ** 2
        for symbolic, numeric in zip(
            discretise(pybamm.InputParameter("R")),
            discretise(pybamm.Scalar(2)),
            strict=True,
        ):
            assert_symbolic_mesh_matches_numeric(symbolic, numeric, y, {"R": 2})

    @pytest.mark.parametrize(
        "coord_sys", ["cartesian", "cylindrical polar", "spherical polar"]
    )
    def test_symbolic_length_mesh_public_matrices(self, coord_sys):
        domains = {"primary": ["particle"], "secondary": ["electrode"]}

        def matrices(radius):
            disc = get_symbolic_length_discretisation_for_testing(
                radius, coord_sys=coord_sys
            )
            method = disc.spatial_methods["particle"]
            return [
                method.gradient_matrix(["particle"], domains),
                method.divergence_matrix(domains),
            ]

        for symbolic, numeric in zip(
            matrices(pybamm.InputParameter("R")),
            matrices(pybamm.Scalar(2)),
            strict=True,
        ):
            assert not symbolic.is_constant()
            np.testing.assert_allclose(
                symbolic.evaluate(inputs={"R": 2}).toarray(),
                numeric.evaluate().toarray(),
                rtol=1e-12,
                atol=1e-12,
            )

    @pytest.mark.parametrize(
        "radius, inputs",
        [(pybamm.Scalar(2), {}), (pybamm.InputParameter("R"), {"R": 2})],
        ids=["numeric", "symbolic"],
    )
    def test_overridden_public_matrices(self, radius, inputs):
        class ScaledFiniteVolume(pybamm.FiniteVolume):
            def gradient_matrix(self, domain, domains):
                return 2 * super().gradient_matrix(domain, domains)

            def divergence_matrix(self, domains):
                return 3 * super().divergence_matrix(domains)

        c = pybamm.Variable(
            "c", "particle", auxiliary_domains={"secondary": "electrode"}
        )
        y = np.linspace(0, 1, 18)[:, np.newaxis] ** 2

        def evaluate(spatial_method):
            disc = get_symbolic_length_discretisation_for_testing(
                radius, spatial_method=spatial_method
            )
            disc.set_variable_slices([c])
            disc.bcs = {
                c: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(0), "Neumann"),
                }
            }
            return [
                disc.process_symbol(expr).evaluate(0, y, inputs=inputs)
                for expr in (pybamm.grad(c), pybamm.div(pybamm.grad(c)))
            ]

        grad, div = evaluate(pybamm.FiniteVolume())
        scaled_grad, scaled_div = evaluate(ScaledFiniteVolume())
        np.testing.assert_allclose(scaled_grad, 2 * grad, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scaled_div, 6 * div, rtol=1e-12, atol=1e-12)

    def test_repeat_vector(self):
        vector = pybamm.InputParameter("a") * pybamm.Vector(np.array([1.0, 2.0]))
        assert pybamm.FiniteVolume._repeat_vector(vector, 1) is vector
        repeated = pybamm.FiniteVolume._repeat_vector(vector, 3)
        assert_constant_matrix_factors(repeated)
        np.testing.assert_array_equal(
            repeated.evaluate(inputs={"a": 2}).ravel(), np.tile([2.0, 4.0], 3)
        )
        constant = pybamm.FiniteVolume._repeat_vector(
            pybamm.Vector(np.array([1.0, 2.0])), 3
        )
        assert isinstance(constant, pybamm.Vector)
        np.testing.assert_array_equal(constant.entries.ravel(), np.tile([1.0, 2.0], 3))

    def test_apply_matrix(self):
        stencil = pybamm.Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
        a = pybamm.InputParameter("a")
        column = a * pybamm.Vector(np.array([1.0, 2.0]))
        vector = pybamm.StateVector(slice(0, 2))
        y = np.array([[5.0], [6.0]])
        values = stencil.entries
        cases = [
            (column, (values * [[2.0], [4.0]]) @ y),
            (pybamm.Transpose(column), (values * [[2.0, 4.0]]) @ y),
            (a * pybamm.Matrix(np.array([[1.0, 2.0]])), (values * [[2.0, 4.0]]) @ y),
            (a, 2 * values @ y),
        ]
        for scale, expected in cases:
            matrix = pybamm.FiniteVolume._scaled_matrix(stencil, scale)
            product = pybamm.FiniteVolume._apply_matrix(matrix, vector)
            assert_constant_matrix_factors(product)
            np.testing.assert_allclose(
                product.evaluate(y=y, inputs={"a": 2}), expected, rtol=1e-15
            )

        # a constant scaling folds into the matrix, and any other matrix multiplies
        # as it is
        constant = pybamm.FiniteVolume._scaled_matrix(stencil, pybamm.Scalar(2))
        assert isinstance(constant, pybamm.Matrix)
        full = pybamm.Multiplication(stencil, a * pybamm.Matrix(np.ones((2, 2))))
        product = pybamm.FiniteVolume._apply_matrix(full, vector)
        assert isinstance(product, pybamm.MatrixMultiplication)
        assert product.left == full
