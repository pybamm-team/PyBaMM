import pybamm
import unittest
from tests import TestCase


class TestSubMesh0D(TestCase):
    def test_exceptions(self):
        position = {"x": {"position": 0}, "y": {"position": 0}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.SubMesh0D(position)

    def test_init(self):
        position = {"x": {"position": 1}}
        generator = pybamm.SubMesh0D
        mesh = generator(position, None)
        mesh.add_ghost_meshes()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
