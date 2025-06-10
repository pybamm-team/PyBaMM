#
# Test for the gradient and divergence in Finite Volumes
#

import numpy as np

import pybamm
from tests import (
    get_mesh_for_testing_2d,
)


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
        linear_x_var = pybamm.Variable("linear_x_var", domain=whole_cell)
        grad_eqn = pybamm.grad(linear_x_var)
        boundary_conditions = {
            linear_x_var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([linear_x_var])
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
        LR, _ = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        submesh_neg = mesh["negative electrode"]
        _, TB_neg = np.meshgrid(submesh_neg.nodes_lr, submesh_neg.nodes_tb)
        linear_y = LR.flatten()
        disc.set_variable_slices([linear_x_var])
        N = pybamm.grad(linear_x_var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            linear_x_var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(submesh.npts),
            rtol=1e-7,
            atol=1e-6,
        )

        # Now do linear in y direction
        linear_y_var = pybamm.Variable("linear_y_var", domain=["negative electrode"])
        grad_eqn = pybamm.grad(linear_y_var)
        boundary_conditions = {
            linear_y_var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([linear_y_var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        linear_y = TB_neg.flatten()
        submesh = mesh["negative electrode"]
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )

    def test_grad_div_shapes_Neumann_bcs(self):
        """
        Test grad and div with Neumann boundary conditions in Cartesian coordinates
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
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )

        ## Test operations on linear x
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        linear_y = LR.flatten()
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb)),
        )

        ## Test operations on linear y
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        linear_y = TB.flatten()
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(1), "Neumann"),
                "bottom": (pybamm.Scalar(1), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb)),
        )

    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs(self):
        """
        Test grad and div with a Dirichlet boundary condition on one side and
        a Neumann boundary conditions on the other side in Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]

        # Test gradient and divergence of a constant
        constant_y = np.ones(submesh.npts_lr * submesh.npts_tb)
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        # grad(1) = 0
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        # div(grad(1)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb)),
        )

        ## Test gradient and divergence of linear x
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        linear_y = LR.flatten()
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        ## grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        ## div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb)),
        )

        ## Test gradient and divergence of linear y
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        linear_y = TB.flatten()
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(1), "Neumann"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        ## grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        ## div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb)),
        )
