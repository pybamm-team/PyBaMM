import pybamm
import unittest


class TestUniform1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        lims = [[0, 1], [0, 1]]
        with self.assertRaises(ValueError):
            pybamm.Uniform1DSubMesh(lims, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
