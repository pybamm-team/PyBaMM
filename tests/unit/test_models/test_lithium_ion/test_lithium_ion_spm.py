#
# Tests for the lithium-ion SPM model
#
import pybamm
import unittest


class TestSPM(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lithium_ion.SPM()
        model.check_well_posedness()

    def test_default_geometry(self):
        model = pybamm.lithium_ion.DFN()
        self.assertTrue(isinstance(model.default_geometry, pybamm.Geometry))
        self.assertTrue("negative particle" in model.default_geometry)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
