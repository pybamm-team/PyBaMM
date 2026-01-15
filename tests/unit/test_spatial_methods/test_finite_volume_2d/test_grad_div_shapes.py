#
# Test for the gradient and divergence in Finite Volumes
#

import numpy as np
import pytest

import pybamm
from tests import DummyDiscretisationClass, get_mesh_for_testing_2d


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
        LR_neg, TB_neg = np.meshgrid(submesh_neg.nodes_lr, submesh_neg.nodes_tb)
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
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
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

        # Now do linear in y direction but with a variable BC
        my_scalar_var_zero = pybamm.Variable("my_scalar_var_zero")
        my_scalar_var_one = pybamm.Variable("my_scalar_var_one")
        linear_y_var = pybamm.Variable("linear_y_var", domain=["negative electrode"])
        disc.set_variable_slices([my_scalar_var_zero, my_scalar_var_one, linear_y_var])
        grad_eqn = pybamm.grad(linear_y_var)

        myclass = DummyDiscretisationClass()
        boundary_conditions = {
            linear_y_var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (my_scalar_var_one, "Dirichlet"),
                "bottom": (my_scalar_var_zero, "Dirichlet"),
            }
        }
        myclass.boundary_conditions = boundary_conditions
        disc.bcs = disc.process_boundary_conditions(myclass)
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        linear_y = np.concatenate([[0], [1], TB_neg.flatten()])
        submesh = mesh["negative electrode"]
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )

        # Now do linear in x direction but with a variable BC
        myclass = DummyDiscretisationClass()
        boundary_conditions = {
            linear_y_var: {
                "left": (my_scalar_var_zero, "Dirichlet"),
                "right": (my_scalar_var_one, "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        myclass.boundary_conditions = boundary_conditions
        disc.bcs = disc.process_boundary_conditions(myclass)
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        linear_y = np.concatenate([[0], [1 / 3], LR_neg.flatten()])
        submesh = mesh["negative electrode"]
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_y).flatten(),
            np.zeros((submesh.npts_tb + 1) * (submesh.npts_lr)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_y).flatten(),
            np.ones((submesh.npts_tb) * (submesh.npts_lr + 1)),
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

        ## Test operations on linear y with input parameter BC's
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        my_scalar_var = pybamm.Variable("my_scalar_var")
        disc.set_variable_slices([my_scalar_var, var])
        linear_y = np.concatenate([[1], LR.flatten()])
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(grad_eqn)

        boundary_conditions = {
            var: {
                "left": (my_scalar_var, "Neumann"),
                "right": (my_scalar_var, "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        myclass = DummyDiscretisationClass()
        myclass.boundary_conditions = boundary_conditions
        disc.bcs = disc.process_boundary_conditions(myclass)
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(
                inputs={"top_bc": 1, "bottom_bc": 1}, y=linear_y
            ).flatten(),
            np.zeros((submesh.npts_tb + 1) * (submesh.npts_lr)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(
                inputs={"top_bc": 1, "bottom_bc": 1}, y=linear_y
            ).flatten(),
            np.ones((submesh.npts_tb) * (submesh.npts_lr + 1)),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(
                inputs={"top_bc": 1, "bottom_bc": 1}, y=linear_y
            ).flatten(),
            np.zeros((submesh.npts_tb) * (submesh.npts_lr)),
        )

        ## Test operations on linear y with input parameter BC's
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        my_scalar_var = pybamm.Variable("my_scalar_var")
        disc.set_variable_slices([my_scalar_var, var])
        linear_y = np.concatenate([[1], TB.flatten()])
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(grad_eqn)

        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (my_scalar_var, "Neumann"),
                "bottom": (my_scalar_var, "Neumann"),
            }
        }
        myclass = DummyDiscretisationClass()
        myclass.boundary_conditions = boundary_conditions
        disc.bcs = disc.process_boundary_conditions(myclass)
        # grad(x) = 1
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(
                inputs={"top_bc": 1, "bottom_bc": 1}, y=linear_y
            ).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(
                inputs={"top_bc": 1, "bottom_bc": 1}, y=linear_y
            ).flatten(),
            np.ones((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )
        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(
                inputs={"top_bc": 1, "bottom_bc": 1}, y=linear_y
            ).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb)),
        )

    @pytest.mark.parametrize(
        "domain",
        [
            ["negative electrode", "separator", "positive electrode"],
            "negative electrode",
        ],
    )
    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs(self, domain):
        """
        Test grad and div with a Dirichlet boundary condition on one side and
        a Neumann boundary conditions on the other side in Cartesian coordinates
        """
        # Create discretisation
        domain = domain
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[domain]

        # Test gradient and divergence of a constant
        constant_y = np.ones(submesh.npts_lr * submesh.npts_tb)
        var = pybamm.Variable("var", domain=domain)
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
        right_val = submesh.edges_lr[-1]
        linear_y = LR.flatten()
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(right_val), "Dirichlet"),
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
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Neumann"),
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

        ## Test gradient and divergence of linear y
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        linear_y = TB.flatten()
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Neumann"),
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

    def test_grad_div_shapes_concatenation(self):
        """
        Test grad and div with concatenation variables using Dirichlet boundary conditions
        """
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume2D(),
            "separator": pybamm.FiniteVolume2D(),
            "positive electrode": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create separate variables for each domain
        var_n = pybamm.Variable("var_n", domain=["negative electrode"])
        var_s = pybamm.Variable("var_s", domain=["separator"])
        var_p = pybamm.Variable("var_p", domain=["positive electrode"])

        # Create concatenation variable
        var_concat = pybamm.concatenation(var_n, var_s, var_p)

        # Get combined submesh
        submesh = mesh[("negative electrode", "separator", "positive electrode")]

        # Test gradient of constant is zero
        # grad(1) = 0
        grad_eqn = pybamm.grad(var_concat)
        boundary_conditions = {
            var_concat: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var_n, var_s, var_p])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        # Create constant values for each domain and concatenate
        submesh_n = mesh["negative electrode"]
        submesh_s = mesh["separator"]
        submesh_p = mesh["positive electrode"]

        constant_n = np.ones(submesh_n.npts)
        constant_s = np.ones(submesh_s.npts)
        constant_p = np.ones(submesh_p.npts)
        constant_y = np.concatenate([constant_n, constant_s, constant_p])

        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, constant_y).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )

        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        linear_x = LR.flatten()

        N = pybamm.grad(var_concat)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var_concat: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions

        # grad(x) = 1 in lr direction, 0 in tb direction
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_x).flatten(),
            np.ones((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_x).flatten(),
            np.zeros((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )

        # div(grad(x)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_x),
            np.zeros_like(submesh.npts),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test operations on linear z
        # Create z-dependent values for each domain and concatenate
        linear_z = TB.flatten()

        boundary_conditions = {
            var_concat: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        # grad(z) = 0 in lr direction, 1 in tb direction
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.lr_field.evaluate(None, linear_z).flatten(),
            np.zeros((submesh.npts_lr + 1) * (submesh.npts_tb)),
        )
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.tb_field.evaluate(None, linear_z).flatten(),
            np.ones((submesh.npts_lr) * (submesh.npts_tb + 1)),
        )

        # div(grad(z)) = 0
        div_eqn_disc = disc.process_symbol(div_eqn)
        np.testing.assert_allclose(
            div_eqn_disc.evaluate(None, linear_z),
            np.zeros_like(submesh.npts),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_laplacian_shapes(self):
        """
        Test Laplacian with concatenation variables using Dirichlet boundary conditions
        """
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume2D(),
            "separator": pybamm.FiniteVolume2D(),
            "positive electrode": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create separate variables for each domain
        var_n = pybamm.Variable("var_n", domain=["negative electrode"])
        var_s = pybamm.Variable("var_s", domain=["separator"])
        var_p = pybamm.Variable("var_p", domain=["positive electrode"])

        # Create concatenation variable
        var_concat = pybamm.concatenation(var_n, var_s, var_p)

        # Get combined submesh
        submesh = mesh[("negative electrode", "separator", "positive electrode")]

        # Test Laplacian of constant is zero
        # laplacian(1) = 0
        laplacian_eqn = pybamm.laplacian(var_concat)
        boundary_conditions = {
            var_concat: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var_n, var_s, var_p])
        laplacian_eqn_disc = disc.process_symbol(laplacian_eqn)

        # Create constant values for each domain and concatenate
        submesh_n = mesh["negative electrode"]
        submesh_s = mesh["separator"]
        submesh_p = mesh["positive electrode"]

        constant_n = np.ones(submesh_n.npts)
        constant_s = np.ones(submesh_s.npts)
        constant_p = np.ones(submesh_p.npts)
        constant_y = np.concatenate([constant_n, constant_s, constant_p])

        np.testing.assert_array_almost_equal(
            laplacian_eqn_disc.evaluate(None, constant_y).flatten(),
            np.zeros(submesh.npts),
        )

        # Test Laplacian of linear x is zero
        # laplacian(x) = 0
        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)
        linear_x = LR.flatten()

        boundary_conditions = {
            var_concat: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions

        laplacian_eqn_disc = disc.process_symbol(laplacian_eqn)
        np.testing.assert_array_almost_equal(
            laplacian_eqn_disc.evaluate(None, linear_x).flatten(),
            np.zeros(submesh.npts),
        )

        # Test Laplacian of linear z is zero
        # laplacian(z) = 0
        linear_z = TB.flatten()

        boundary_conditions = {
            var_concat: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        laplacian_eqn_disc = disc.process_symbol(laplacian_eqn)
        np.testing.assert_array_almost_equal(
            laplacian_eqn_disc.evaluate(None, linear_z).flatten(),
            np.zeros(submesh.npts),
        )

    def test_internal_neumann_condition(self):
        """
        Test internal Neumann conditions (flux continuity) between domains
        """
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume2D(),
            "separator": pybamm.FiniteVolume2D(),
            "positive electrode": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create variables for adjacent domains
        var_n = pybamm.Variable("var_n", domain=["negative electrode"])
        var_s = pybamm.Variable("var_s", domain=["separator"])
        var_p = pybamm.Variable("var_p", domain=["positive electrode"])

        disc.set_variable_slices([var_n, var_s, var_p])

        # Get submeshes
        submesh_n = mesh["negative electrode"]
        submesh_s = mesh["separator"]
        submesh_p = mesh["positive electrode"]

        # Test 1: Internal Neumann condition between negative electrode and separator
        # Create a linear function across both domains
        LR_n, _TB_n = np.meshgrid(submesh_n.nodes_lr, submesh_n.nodes_tb)
        LR_s, _TB_s = np.meshgrid(submesh_s.nodes_lr, submesh_s.nodes_tb)
        LR_p, _TB_p = np.meshgrid(submesh_p.nodes_lr, submesh_p.nodes_tb)

        # Linear x function - should have continuous gradient across boundary
        linear_x_n = LR_n.flatten()
        linear_x_s = LR_s.flatten()
        linear_x_p = LR_p.flatten()

        # Process the variables
        var_n_disc = disc.process_symbol(var_n)
        var_s_disc = disc.process_symbol(var_s)
        var_p_disc = disc.process_symbol(var_p)

        # Get the spatial method to test internal_neumann_condition directly
        spatial_method = spatial_methods["negative electrode"]
        spatial_method.build(mesh)

        # Test internal Neumann condition between negative electrode and separator
        internal_neumann = spatial_method.internal_neumann_condition(
            var_n_disc, var_s_disc, submesh_n, submesh_s
        )

        # For a linear x function, the internal Neumann condition computes:
        # (first_value_of_separator - last_value_of_neg_electrode) / dx
        # This should equal the slope of the linear function (which is 1)
        result = internal_neumann.evaluate(
            None, np.concatenate([linear_x_n, linear_x_s, linear_x_p])
        )

        # For a linear x function, this should equal 1 (the slope)
        # The exact value depends on the mesh spacing
        dx = submesh_s.nodes_lr[0] - submesh_n.nodes_lr[-1]  # spacing across boundary
        expected_slope = 1.0  # slope of linear x function

        np.testing.assert_allclose(
            result.flatten(),
            expected_slope
            * np.ones(submesh_n.npts_tb),  # One value per tb node at the boundary
            rtol=1e-6,
        )

        # Test 2: Internal Neumann condition between separator and positive electrode
        # Test internal Neumann condition between separator and positive electrode
        internal_neumann_sp = spatial_method.internal_neumann_condition(
            var_s_disc, var_p_disc, submesh_s, submesh_p
        )

        result_sp = internal_neumann_sp.evaluate(
            None, np.concatenate([linear_x_n, linear_x_s, linear_x_p])
        )

        # Should also equal the slope (1) for linear x function
        np.testing.assert_allclose(
            result_sp.flatten(),
            expected_slope
            * np.ones(submesh_s.npts_tb),  # One value per tb node at the boundary
            rtol=1e-6,
        )

        # Test 3: Test with constant function - should give zero
        # Create constant functions with the same value in each domain
        constant_n = np.ones(submesh_n.npts)  # value = 1 in negative electrode
        constant_s = np.ones(submesh_s.npts)  # value = 1 in separator
        constant_p = np.ones(submesh_p.npts)  # value = 1 in positive electrode

        # This should give zero since there's no jump across the boundary
        result_constant = internal_neumann.evaluate(
            None, np.concatenate([constant_n, constant_s, constant_p])
        )

        # The result should be zero for constant function (no jump)
        np.testing.assert_allclose(
            result_constant.flatten(),
            np.zeros(submesh_n.npts_tb),
            rtol=1e-10,
            atol=1e-12,
        )

        # Test 4: Test with discontinuous function to verify expected jump
        # Create a step function with different values in each domain
        constant_n_step = np.ones(submesh_n.npts)  # value = 1 in negative electrode
        constant_s_step = 3 * np.ones(submesh_s.npts)  # value = 3 in separator
        constant_p_step = 3 * np.ones(submesh_p.npts)  # value = 3 in positive electrode

        # This should give the jump divided by dx
        result_discontinuous = internal_neumann.evaluate(
            None, np.concatenate([constant_n_step, constant_s_step, constant_p_step])
        )

        # The result should be (3 - 1) / dx = 2 / dx
        expected_jump = (3 - 1) / dx  # (right_value - left_value) / dx

        # Check that we get the expected jump
        np.testing.assert_allclose(
            result_discontinuous.flatten(),
            expected_jump * np.ones(submesh_n.npts_tb),
            rtol=1e-6,
        )

    def test_grad_direction_error(self):
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        disc_var = disc.set_variable_slices([var])
        symbol = pybamm.Gradient(var)
        spatial_method = pybamm.FiniteVolume2D()
        spatial_method.build(mesh)
        with pytest.raises(ValueError, match=r"Direction asdf not supported"):
            spatial_method._gradient(symbol, disc_var, None, direction="asdf")
