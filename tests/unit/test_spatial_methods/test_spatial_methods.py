#
# Test for the base Spatial Method class
#
import pybamm
from tests import get_mesh_for_testing

import unittest


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
            spatial_method.integral(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.indefinite_integral(None, None, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
