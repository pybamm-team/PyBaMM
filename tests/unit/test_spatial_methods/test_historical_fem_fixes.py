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
        """
        pytest.importorskip("skfem")
        from tests import get_unit_3d_mesh_for_testing

        mesh = get_unit_3d_mesh_for_testing(h=0.5)

        spatial_method = pybamm.ScikitFiniteElement3D()
        spatial_method.build(mesh)

        scalar = pybamm.Scalar(1.0)
        scalar.domains = {"primary": ["current collector"]}

        grad_result = spatial_method.gradient(scalar, scalar, {})

        assert isinstance(grad_result, pybamm.Concatenation)

        for child in grad_result.children:
            assert isinstance(child, pybamm.Vector)
            child_vals = child.entries
            np.testing.assert_allclose(child_vals, 0.0, atol=1e-15)
