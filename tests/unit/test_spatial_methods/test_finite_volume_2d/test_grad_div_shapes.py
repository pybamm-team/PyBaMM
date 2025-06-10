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
