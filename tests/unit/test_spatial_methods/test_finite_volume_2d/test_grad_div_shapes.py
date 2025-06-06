#
# Test for the gradient and divergence in Finite Volumes
#

import pybamm
from tests import (
    get_mesh_for_testing_2d,
)
import numpy as np


class TestFiniteVolume2DGradDiv:
    def test_grad_div_shapes_Dirichlet_bcs(self):
        """
        Test grad and div with Dirichlet boundary conditions in Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient of constant is zero
        # grad(1) = 0
        constant_y = np.ones(submesh.npts_lr * submesh.npts_tb)
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_equal(
            grad_eqn_disc.lr_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_equal(
            grad_eqn_disc.tb_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )

        # Test operations on linear x
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        submesh_neg = mesh["negative electrode"]
        submesh_sep = mesh["separator"]
        submesh_pos = mesh["positive electrode"]
        LR_neg, TB_neg = np.meshgrid(submesh_neg.nodes_lr, submesh_neg.nodes_tb)
        LR_sep, TB_sep = np.meshgrid(submesh_sep.nodes_lr, submesh_sep.nodes_tb)
        LR_pos, TB_pos = np.meshgrid(submesh_pos.nodes_lr, submesh_pos.nodes_tb)
        linear_y_neg = LR_neg.flatten()
        linear_y_sep = LR_sep.flatten()
        linear_y_pos = LR_pos.flatten()
        linear_y = LR.flatten()
        disc.set_variable_slices([var])
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_allclose(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr + 1) * (submesh.npts_tb)),
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.nodes[:, np.newaxis]),
            rtol=1e-7,
            atol=1e-6,
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

    def test_grad_div_shapes_mixed_domain(self):
        # Create discretisation
        raise NotImplementedError
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
