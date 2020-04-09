#
# Test for the base Spatial Method class
#
import numpy as np
import pybamm
import unittest
from tests import get_mesh_for_testing, get_1p1d_mesh_for_testing


class TestZeroDimensionalSpatialMethod(unittest.TestCase):
    def test_identity_ops(self):
        test_mesh = np.array([1, 2, 3])
        spatial_method = pybamm.ZeroDimensionalSpatialMethod()
        spatial_method.build(test_mesh)
        np.testing.assert_array_equal(spatial_method._mesh, test_mesh)

        a = pybamm.Symbol("a")
        self.assertEqual(a, spatial_method.integral(None, a))
        self.assertEqual(a, spatial_method.indefinite_integral(None, a))
        self.assertEqual(a, spatial_method.boundary_value_or_flux(None, a))

        mass_matrix = spatial_method.mass_matrix(None, None)
        self.assertIsInstance(mass_matrix, pybamm.Matrix)
        self.assertEqual(mass_matrix.shape, (1, 1))
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
            self.assertIsInstance(var_disc, pybamm.Vector)
            np.testing.assert_array_equal(
                var_disc.evaluate()[:, 0], mesh.combine_submeshes(*var.domain)[0].nodes
            )

        # edges
        x1_edge = pybamm.SpatialVariableEdge("x", ["negative electrode"])
        x2_edge = pybamm.SpatialVariableEdge("x", ["negative electrode", "separator"])
        r_edge = pybamm.SpatialVariableEdge("r", ["negative particle"])
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
        spatial_method = pybamm.SpatialMethod()
        spatial_method.build(mesh)
        with self.assertRaisesRegex(TypeError, "Cannot process BoundaryGradient"):
            spatial_method.boundary_value_or_flux(symbol, child)

        mesh = get_1p1d_mesh_for_testing()
        spatial_method = pybamm.SpatialMethod()
        spatial_method.build(mesh)
        with self.assertRaisesRegex(NotImplementedError, "Cannot process 2D symbol"):
            spatial_method.boundary_value_or_flux(symbol, child)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
