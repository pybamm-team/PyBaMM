import numpy as np
import pybamm
from tests import get_mesh_for_testing_3d


class TestGradDivShapes3D:
    def test_grad_div_shapes_Dirichlet_bcs_3d(self):
        """
        Test grad and div with Dirichlet boundary conditions in 3D Cartesian coordinates
        """
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

        # Test gradient of constant is zero
        # grad(1) = 0
        constant_y = np.ones((n_x * n_y * n_z, 1))
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "front": (pybamm.Scalar(1), "Dirichlet"),
                "back": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        # Expected: concatenated zero vectors for all three components
        zeros_x = np.zeros(((n_x + 1) * n_y * n_z, 1))
        zeros_y = np.zeros((n_x * (n_y + 1) * n_z, 1))
        zeros_z = np.zeros((n_x * n_y * (n_z + 1), 1))
        expected_zeros = np.concatenate([zeros_x, zeros_y, zeros_z])

        result = grad_eqn_disc.evaluate(None, constant_y)
        np.testing.assert_array_equal(result, expected_zeros)

        # Test operations on linear x
        # For 3D, use x-coordinate of nodes as the linear function
        coords_x = submesh.nodes[:, 0:1]  # Extract x-coordinates as column vector
        print("coords_x", coords_x.shape)
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "front": (pybamm.Scalar(0), "Dirichlet"),
                "back": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        # grad(x) should be [1, 0, 0] - only x-component is 1, y and z components are 0
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        ones_x = np.ones(((n_x + 1) * n_y * n_z, 1))
        zeros_y = np.zeros((n_x * (n_y + 1) * n_z, 1))
        zeros_z = np.zeros((n_x * n_y * (n_z + 1), 1))
        expected_grad = np.concatenate([ones_x, zeros_y, zeros_z])

        result = grad_eqn_disc.evaluate(None, coords_x)
        np.testing.assert_allclose(result, expected_grad, rtol=1e-7, atol=1e-6)

        # div(grad(x)) = 0 (second derivative of linear function is zero)
        div_eqn_disc = disc.process_symbol(div_eqn)
        result = div_eqn_disc.evaluate(None, coords_x)
        np.testing.assert_allclose(
            result, np.zeros((n_x * n_y * n_z, 1)), rtol=1e-7, atol=1e-6
        )

    def test_grad_div_shapes_Neumann_bcs_3d(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
        constant_y = np.ones((n_x * n_y * n_z, 1))
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        zeros = np.zeros(
            ((n_x + 1) * n_y * n_z + n_x * (n_y + 1) * n_z + n_x * n_y * (n_z + 1), 1)
        )
        result = grad_eqn_disc.evaluate(None, constant_y)
        np.testing.assert_array_equal(result, zeros)
        coords = submesh.nodes[:, 0]
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        ones = np.ones(
            ((n_x + 1) * n_y * n_z + n_x * (n_y + 1) * n_z + n_x * n_y * (n_z + 1), 1)
        )
        result = grad_eqn_disc.evaluate(None, coords[:, np.newaxis])
        np.testing.assert_allclose(result, ones, rtol=1e-7, atol=1e-6)
        div_eqn = pybamm.div(pybamm.grad(var))
        div_eqn_disc = disc.process_symbol(div_eqn)
        result = div_eqn_disc.evaluate(None, coords[:, np.newaxis])
        np.testing.assert_allclose(
            result, np.zeros((n_x * n_y * n_z, 1)), rtol=1e-7, atol=1e-6
        )

    def test_grad_div_shapes_Dirichlet_and_Neumann_bcs_3d(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
        constant_y = np.ones((n_x * n_y * n_z, 1))
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(pybamm.grad(var))
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        print("grad_eqn_disc", grad_eqn_disc.shape)
        zeros = np.zeros(
            ((n_x + 1) * n_y * n_z + n_x * (n_y + 1) * n_z + n_x * n_y * (n_z + 1), 1)
        )
        result = grad_eqn_disc.evaluate(None, constant_y)
        np.testing.assert_array_equal(result, zeros)
        div_eqn_disc = disc.process_symbol(div_eqn)
        result = div_eqn_disc.evaluate(None, constant_y)
        np.testing.assert_allclose(
            result, np.zeros((n_x * n_y * n_z, 1)), rtol=1e-7, atol=1e-6
        )
        coords = submesh.nodes[:, 0]
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "front": (pybamm.Scalar(0), "Neumann"),
                "back": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        print("grad_eqn_disc", grad_eqn_disc.shape)
        ones = np.ones(
            ((n_x + 1) * n_y * n_z + n_x * (n_y + 1) * n_z + n_x * n_y * (n_z + 1), 1)
        )
        result = grad_eqn_disc.evaluate(None, coords[:, np.newaxis])
        np.testing.assert_allclose(result, ones, rtol=1e-7, atol=1e-6)
        div_eqn_disc = disc.process_symbol(div_eqn)
        result = div_eqn_disc.evaluate(None, coords[:, np.newaxis])
        np.testing.assert_allclose(
            result, np.zeros((n_x * n_y * n_z, 1)), rtol=1e-7, atol=1e-6
        )
