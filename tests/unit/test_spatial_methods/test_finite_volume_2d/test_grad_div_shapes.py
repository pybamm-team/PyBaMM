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

        # Test operations on linear x
        # Create x-dependent values for each domain and concatenate
        LR_n, TB_n = np.meshgrid(submesh_n.nodes_lr, submesh_n.nodes_tb)
        LR_s, TB_s = np.meshgrid(submesh_s.nodes_lr, submesh_s.nodes_tb)
        LR_p, TB_p = np.meshgrid(submesh_p.nodes_lr, submesh_p.nodes_tb)

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
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
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
        np.testing.assert_allclose(
            laplacian_eqn_disc.evaluate(None, linear_x).flatten(),
            np.zeros(submesh.npts),
            rtol=1e-7,
            atol=1e-6,
        )

        # Test Laplacian of linear z is zero
        # laplacian(z) = 0
        linear_z = TB.flatten()

        boundary_conditions = {
            var_concat: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        laplacian_eqn_disc = disc.process_symbol(laplacian_eqn)
        np.testing.assert_allclose(
            laplacian_eqn_disc.evaluate(None, linear_z).flatten(),
            np.zeros(submesh.npts),
            rtol=1e-7,
            atol=1e-6,
        )
