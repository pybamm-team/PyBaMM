import pybamm
import unittest
import numpy as np


class TestSubMesh1D(unittest.TestCase):
    def test_tabs(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0}, "positive": {"z_centre": 1}}
        mesh = pybamm.one_dimensional_meshes.SubMesh1D(edges, None, tabs=tabs)
        self.assertEqual(mesh.tabs["negative tab"], "left")
        self.assertEqual(mesh.tabs["positive tab"], "right")

    def test_exceptions(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0.2}, "positive": {"z_centre": 1}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.one_dimensional_meshes.SubMesh1D(edges, None, tabs=tabs)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
