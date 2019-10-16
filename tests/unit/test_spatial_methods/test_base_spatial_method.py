#
# Test for the base Spatial Method class
#
import numpy as np
import pybamm
import unittest
from tests import get_mesh_for_testing, get_1p1d_mesh_for_testing


class TestSpatialMethod(unittest.TestCase):
    def test_basics(self):
        mesh = get_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod(mesh)
        self.assertEqual(spatial_method.mesh, mesh)
        with self.assertRaises(NotImplementedError):
            spatial_method.gradient(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.divergence(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.laplacian(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.gradient_squared(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.integral(None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.indefinite_integral(None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.boundary_integral(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.delta_function(None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.internal_neumann_condition(None, None, None, None)

    def test_discretise_spatial_variable(self):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod(mesh)

        # centre
        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        r = pybamm.SpatialVariable("r", ["negative particle"])
        for var in [x1, x2, r]:
            var_disc = spatial_method.spatial_variable(var)
            self.assertIsInstance(var_disc, pybamm.Vector)
            np.testing.assert_array_equal(
                var_disc.evaluate()[:, 0], mesh.combine_submeshes(*var.domain)[0].nodes
            )

        # edges
        x1_edge = pybamm.SpatialVariable("x_edge", ["negative electrode"])
        x2_edge = pybamm.SpatialVariable("x_edge", ["negative electrode", "separator"])
        r_edge = pybamm.SpatialVariable("r_edge", ["negative particle"])
        for var in [x1_edge, x2_edge, r_edge]:
            var_disc = spatial_method.spatial_variable(var)
            self.assertIsInstance(var_disc, pybamm.Vector)
            np.testing.assert_array_equal(
                var_disc.evaluate()[:, 0], mesh.combine_submeshes(*var.domain)[0].edges
            )

    def test_broadcast_checks(self):
        child = pybamm.Symbol("sym", domain=["negative electrode"])
        symbol = pybamm.BoundaryGradient(child, "left")
        mesh = get_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod(mesh)
        with self.assertRaisesRegex(TypeError, "Cannot process BoundaryGradient"):
            spatial_method.boundary_value_or_flux(symbol, child)

        mesh = get_1p1d_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod(mesh)
        with self.assertRaisesRegex(NotImplementedError, "Cannot process 2D symbol"):
            spatial_method.boundary_value_or_flux(symbol, child)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
