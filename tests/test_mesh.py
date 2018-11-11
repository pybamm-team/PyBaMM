from pybamm.mesh import *
from pybamm.parameters import *

import numpy as np
import unittest


class TestMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = Parameters()
        mesh = Mesh(param, 50)
        self.assertEqual(mesh.nn + mesh.ns + mesh.np, mesh.n)
        self.assertEqual(mesh.x[-1], 1)
        self.assertEqual(len(mesh.x), len(mesh.xc) + 1)
        self.assertAlmostEqual(
            np.linalg.norm(mesh.xc - np.concatenate([mesh.xcn, mesh.xcs, mesh.xcp])), 0
        )


if __name__ == "__main__":
    unittest.main()
