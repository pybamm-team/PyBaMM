#
# Test for the mesh class
#
import pybamm

import numpy as np
import unittest


class TestMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.Parameters()
        mesh = pybamm.Mesh(param, 50)
        self.assertEqual(mesh.nn + mesh.ns + mesh.np, mesh.n)
        self.assertEqual(mesh.x[-1], 1)
        self.assertEqual(len(mesh.x), len(mesh.xc) + 1)
        self.assertAlmostEqual(
            np.linalg.norm(mesh.xc - np.concatenate([mesh.xcn, mesh.xcs, mesh.xcp])), 0
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
