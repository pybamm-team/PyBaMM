"""
Regression tests for historical spatial method bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestScikitFEM3DFixes:
    """Guards for 3D FEM gradient calculation bug fixes."""

    def test_gradient_of_scalar_returns_zeros(self):
        """
        Guards against: PR #5143 - Fix non-deterministic ShapeError in 3D FEM
        gradient method

        The bug was that when the discretised symbol was a pybamm.Scalar,
        the gradient calculation would fail with a ShapeError. The fix adds
        an explicit check for Scalar inputs and returns a zero gradient.

        This test directly calls the gradient method with a Scalar input
        to verify it returns zeros without raising a ShapeError.
        """
        pytest.importorskip("skfem")
        from tests import get_unit_3d_mesh_for_testing

        # Set up a proper 3D mesh using the test helper
        mesh = get_unit_3d_mesh_for_testing(h=0.5)

        # Create spatial method and build it on the mesh
        spatial_method = pybamm.ScikitFiniteElement3D()
        spatial_method.build(mesh)

        # Create a Scalar with the correct domain
        scalar = pybamm.Scalar(1.0)
        scalar.domains = {"primary": ["current collector"]}

        # The gradient method should handle Scalar inputs
        # Before the fix, this would raise ShapeError due to matrix/scalar mismatch
        grad_result = spatial_method.gradient(scalar, scalar, {})

        # Result should be a Concatenation of zeros
        assert isinstance(grad_result, pybamm.Concatenation)

        # Verify each child of the concatenation is a Vector of zeros
        # The gradient returns [grad_x, grad_y, grad_z] components
        for child in grad_result.children:
            assert isinstance(child, pybamm.Vector)
            child_vals = child.entries
            np.testing.assert_allclose(child_vals, 0.0, atol=1e-15)


class TestFEMConsistency:
    """Tests for FEM spatial method consistency."""

    def test_finite_volume_gradient_matches_analytic(self):
        """
        Verify finite volume gradient calculation matches analytic solution.
        """
        # Simple 1D test - gradient of linear function should be constant
        x = pybamm.SpatialVariable("x", domain="test")
        geometry = {"test": {x: {"min": 0, "max": 1}}}

        submesh_types = {"test": pybamm.Uniform1DSubMesh}
        var_pts = {x: 50}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods = {"test": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create a linear function c(x) = x
        var = pybamm.Variable("c", domain="test")
        grad_c = pybamm.Gradient(var)

        disc.set_variable_slices([var])
        disc_grad = disc.process_symbol(grad_c)

        # Evaluate with y = x (linear)
        x_nodes = mesh["test"].nodes
        y = x_nodes.flatten()

        result = disc_grad.evaluate(y=y)

        # Gradient of x with respect to x should be 1
        # (edges, so one less point)
        expected = np.ones_like(result)
        np.testing.assert_allclose(result, expected, rtol=0.1)
