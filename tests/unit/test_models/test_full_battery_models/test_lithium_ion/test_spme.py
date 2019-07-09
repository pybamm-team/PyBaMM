#
# Tests for the lithium-ion SPMe model
#
import pybamm
import unittest


class TestSPMe(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" in model.default_geometry)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
