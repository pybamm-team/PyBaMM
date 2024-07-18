#
# Test for the base Spatial Method class
#
import numpy as np
import pybamm
from tests import get_mesh_for_testing, get_discretisation_for_testing


class TestZeroDimensionalSpatialMethod:
    def test_identity_ops(self):
        test_mesh = np.array([1, 2, 3])
        spatial_method = pybamm.ZeroDimensionalSpatialMethod()
        spatial_method.build(test_mesh)
        np.testing.assert_array_equal(spatial_method._mesh, test_mesh)

        a = pybamm.Symbol("a")
        assert a == spatial_method.integral(None, a, "primary")
        assert a == spatial_method.indefinite_integral(None, a, "forward")
        assert a == spatial_method.boundary_value_or_flux(None, a)
        assert (-a) == spatial_method.indefinite_integral(None, a, "backward")

        mass_matrix = spatial_method.mass_matrix(None, None)
        assert isinstance(mass_matrix, pybamm.Matrix)
        assert mass_matrix.shape == (1, 1)
        np.testing.assert_array_equal(mass_matrix.entries, 1)

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

    def test_averages(self):
        # create discretisation
        disc = get_discretisation_for_testing(
            cc_method=pybamm.ZeroDimensionalSpatialMethod
        )
        # create and discretise variable
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        var_disc = disc.process_symbol(var)
        # check average returns the same value
        y = np.array([1])
        for expression in [pybamm.z_average(var), pybamm.yz_average(var)]:
            expr_disc = disc.process_symbol(expression)
            np.testing.assert_array_equal(
                var_disc.evaluate(y=y), expr_disc.evaluate(y=y)
            )
