import pybamm
import unittest


class TestSubMesh0D(unittest.TestCase):
    def test_exceptions(self):
        position = {"x": 0, "y": 0}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.SubMesh0D(position)

    def test_init(self):
        position = {"x": 1}
        generator = pybamm.MeshGenerator0D()
        mesh = generator(position)
        mesh.add_ghost_meshes()


class TestMeshGenerator0D(unittest.TestCase):
    def test_exceptions(self):
        generator = pybamm.MeshGenerator1D("bad mesh")
        with self.assertRaisesRegex(pybamm.GeometryError, "Submesh"):
            generator(None, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
