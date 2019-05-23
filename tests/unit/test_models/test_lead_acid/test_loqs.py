#
# Tests for the lead-acid LOQS model
#
import pybamm
import unittest


class TestLeadAcidLOQS(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.LOQS()
        model.check_well_posedness()

    def test_default_geometry(self):
        model = pybamm.lead_acid.LOQS()
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_default_spatial_methods(self):
        model = pybamm.lead_acid.LOQS()
        self.assertIsInstance(model.default_spatial_methods, dict)
        self.assertTrue("negative particle" not in model.default_geometry)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
