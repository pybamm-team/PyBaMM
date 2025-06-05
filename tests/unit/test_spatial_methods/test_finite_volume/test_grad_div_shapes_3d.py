import numpy as np
import pybamm
from tests import get_mesh_for_testing_3d


class TestGradDivShapes3D:
    def test_grad_div_shapes_Dirichlet_bcs_3d(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing_3d(xpts=4, ypts=3, zpts=2)
        spatial_methods = {"macroscale": pybamm.FiniteVolume3D()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
        constant_y = np.ones(n_x * n_y * n_z)
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var: {
                ("x", "left"): (pybamm.Scalar(1), "Dirichlet"),
                ("x", "right"): (pybamm.Scalar(1), "Dirichlet"),
                ("y", "front"): (pybamm.Scalar(1), "Dirichlet"),
                ("y", "back"): (pybamm.Scalar(1), "Dirichlet"),
                ("z", "bottom"): (pybamm.Scalar(1), "Dirichlet"),
                ("z", "top"): (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        print("grad_eqn_disc", grad_eqn_disc)
        zeros_x = np.zeros((n_x + 1) * n_y * n_z)
        zeros_y = np.zeros(n_x * (n_y + 1) * n_z)
        zeros_z = np.zeros(n_x * n_y * (n_z + 1))
        result = grad_eqn_disc.evaluate(None, constant_y).flatten()
        np.testing.assert_array_equal(
            result, np.concatenate([zeros_x, zeros_y, zeros_z])
        )
        coords = submesh.nodes[:, 0]
        boundary_conditions = {
            var: {
                ("x", "left"): (pybamm.Scalar(0), "Dirichlet"),
                ("x", "right"): (pybamm.Scalar(1), "Dirichlet"),
                ("y", "front"): (pybamm.Scalar(0), "Neumann"),
                ("y", "back"): (pybamm.Scalar(0), "Neumann"),
                ("z", "bottom"): (pybamm.Scalar(0), "Neumann"),
                ("z", "top"): (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        ones_x = np.ones((n_x + 1) * n_y * n_z)
        zeros_y = np.zeros(n_x * (n_y + 1) * n_z)
        zeros_z = np.zeros(n_x * n_y * (n_z + 1))
        result = grad_eqn_disc.evaluate(None, coords).flatten()
        np.testing.assert_allclose(
            result, np.concatenate([ones_x, zeros_y, zeros_z]), rtol=1e-7, atol=1e-6
        )
        div_eqn = pybamm.div(pybamm.grad(var))
        div_eqn_disc = disc.process_symbol(div_eqn)
        result = div_eqn_disc.evaluate(None, coords)
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
                ("x", "left"): (pybamm.Scalar(0), "Neumann"),
                ("x", "right"): (pybamm.Scalar(0), "Neumann"),
                ("y", "front"): (pybamm.Scalar(0), "Neumann"),
                ("y", "back"): (pybamm.Scalar(0), "Neumann"),
                ("z", "bottom"): (pybamm.Scalar(0), "Neumann"),
                ("z", "top"): (pybamm.Scalar(0), "Neumann"),
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
                ("x", "left"): (pybamm.Scalar(1), "Neumann"),
                ("x", "right"): (pybamm.Scalar(1), "Neumann"),
                ("y", "front"): (pybamm.Scalar(0), "Neumann"),
                ("y", "back"): (pybamm.Scalar(0), "Neumann"),
                ("z", "bottom"): (pybamm.Scalar(0), "Neumann"),
                ("z", "top"): (pybamm.Scalar(0), "Neumann"),
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
                ("x", "left"): (pybamm.Scalar(1), "Dirichlet"),
                ("x", "right"): (pybamm.Scalar(0), "Neumann"),
                ("y", "front"): (pybamm.Scalar(0), "Neumann"),
                ("y", "back"): (pybamm.Scalar(0), "Neumann"),
                ("z", "bottom"): (pybamm.Scalar(0), "Neumann"),
                ("z", "top"): (pybamm.Scalar(0), "Neumann"),
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
                ("x", "left"): (pybamm.Scalar(1), "Neumann"),
                ("x", "right"): (pybamm.Scalar(1), "Dirichlet"),
                ("y", "front"): (pybamm.Scalar(0), "Neumann"),
                ("y", "back"): (pybamm.Scalar(0), "Neumann"),
                ("z", "bottom"): (pybamm.Scalar(0), "Neumann"),
                ("z", "top"): (pybamm.Scalar(0), "Neumann"),
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
