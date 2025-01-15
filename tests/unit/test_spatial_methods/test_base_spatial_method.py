#
# Test for the base Spatial Method class
#

import pytest
import numpy as np
import pybamm
from tests import (
    get_mesh_for_testing,
    get_1p1d_mesh_for_testing,
    get_size_distribution_mesh_for_testing,
)


class TestSpatialMethod:
    def test_basics(self):
        mesh = get_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod()
        spatial_method.build(mesh)
        assert spatial_method.mesh == mesh
        with pytest.raises(NotImplementedError):
            spatial_method.gradient(None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.divergence(None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.laplacian(None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.gradient_squared(None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.integral(None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.indefinite_integral(None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.boundary_integral(None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.delta_function(None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.internal_neumann_condition(None, None, None, None)
        with pytest.raises(NotImplementedError):
            spatial_method.evaluate_at(None, None, None)

    def test_get_auxiliary_domain_repeats(self):
        # Test the method to read number of repeats from auxiliary domains
        mesh = get_size_distribution_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod()
        spatial_method.build(mesh)

        # No auxiliary domains
        repeats = spatial_method._get_auxiliary_domain_repeats({})
        assert repeats == 1

        # Just secondary domain
        repeats = spatial_method._get_auxiliary_domain_repeats(
            {"secondary": ["negative electrode"]}
        )
        assert repeats == mesh["negative electrode"].npts

        repeats = spatial_method._get_auxiliary_domain_repeats(
            {"secondary": ["negative electrode", "separator"]}
        )
        assert repeats == mesh["negative electrode"].npts + mesh["separator"].npts

        # With tertiary domain
        repeats = spatial_method._get_auxiliary_domain_repeats(
            {
                "secondary": ["negative electrode", "separator"],
                "tertiary": ["current collector"],
            }
        )
        assert (
            repeats
            == (mesh["negative electrode"].npts + mesh["separator"].npts)
            * mesh["current collector"].npts
        )

        # Just tertiary domain
        repeats = spatial_method._get_auxiliary_domain_repeats(
            {"tertiary": ["current collector"]},
        )
        assert repeats == mesh["current collector"].npts

        # With quaternary domain
        repeats = spatial_method._get_auxiliary_domain_repeats(
            {
                "secondary": ["negative particle size"],
                "tertiary": ["negative electrode", "separator"],
                "quaternary": ["current collector"],
            }
        )
        assert (
            repeats
            == mesh["negative particle size"].npts
            * (mesh["negative electrode"].npts + mesh["separator"].npts)
            * mesh["current collector"].npts
        )

    def test_discretise_spatial_variable(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod()
        spatial_method.build(mesh)

        # centre
        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        r = pybamm.SpatialVariable("r", ["negative particle"])
        for var in [x1, x2, r]:
            var_disc = spatial_method.spatial_variable(var)
            assert isinstance(var_disc, pybamm.Vector)
            np.testing.assert_array_equal(
                var_disc.evaluate()[:, 0], mesh[var.domain].nodes
            )

        # edges
        x1_edge = pybamm.SpatialVariableEdge("x", ["negative electrode"])
        x2_edge = pybamm.SpatialVariableEdge("x", ["negative electrode", "separator"])
        r_edge = pybamm.SpatialVariableEdge("r", ["negative particle"])
        for var in [x1_edge, x2_edge, r_edge]:
            var_disc = spatial_method.spatial_variable(var)
            assert isinstance(var_disc, pybamm.Vector)
            np.testing.assert_array_equal(
                var_disc.evaluate()[:, 0], mesh[var.domain].edges
            )

    def test_boundary_value_checks(self):
        child = pybamm.Symbol("sym", domain=["negative electrode"])
        symbol = pybamm.BoundaryGradient(child, "left")
        mesh = get_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod()
        spatial_method.build(mesh)
        with pytest.raises(TypeError, match="Cannot process BoundaryGradient"):
            spatial_method.boundary_value_or_flux(symbol, child)

        # test also symbol "right"
        symbol = pybamm.BoundaryGradient(child, "right")
        with pytest.raises(TypeError, match="Cannot process BoundaryGradient"):
            spatial_method.boundary_value_or_flux(symbol, child)

        mesh = get_1p1d_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod()
        spatial_method.build(mesh)
        child = pybamm.Symbol(
            "sym",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        symbol = pybamm.BoundaryGradient(child, "left")
        with pytest.raises(NotImplementedError, match="Cannot process 2D symbol"):
            spatial_method.boundary_value_or_flux(symbol, child)
