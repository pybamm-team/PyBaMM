#
# Test for the Finite Volume Mesh class
#
import pybamm

import numpy as np
import unittest


class TestFiniteVolumeMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues()
        mesh = pybamm.FiniteVolumesMacroMesh(param, 50)
        self.assertEqual(mesh.x.edges[-1], 1)
        self.assertEqual(len(mesh.x.edges), len(mesh.x.centres) + 1)
        self.assertAlmostEqual(
            np.linalg.norm(
                mesh.x.centres
                - np.concatenate([mesh.xn.centres, mesh.xs.centres, mesh.xp.centres])
            ),
            0,
        )

    def test_mesh_sizes(self):
        param = pybamm.ParameterValues()
        mesh = pybamm.FiniteVolumesMacroMesh(param, 50)
        self.assertEqual(mesh.nn + (mesh.ns - 2) + mesh.np, mesh.n)
        self.assertEqual(mesh.xn.npts, mesh.nn - 1)
        self.assertEqual(mesh.xs.npts, mesh.ns - 1)
        self.assertEqual(mesh.xp.npts, mesh.np - 1)
        self.assertEqual(mesh.x.npts, mesh.n - 1)

        self.assertEqual(len(mesh.xn.edges), mesh.nn)
        self.assertEqual(len(mesh.xs.edges), mesh.ns)
        self.assertEqual(len(mesh.xp.edges), mesh.np)
        self.assertEqual(len(mesh.x.edges), mesh.n)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
