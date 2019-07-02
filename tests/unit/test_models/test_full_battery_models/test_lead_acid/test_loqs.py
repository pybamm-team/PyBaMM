#
# Tests for the lead-acid LOQS model
#
import pybamm
import unittest


class TestLeadAcidLOQS(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"thermal": None, "convection": True}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": None}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_default_spatial_methods(self):
        options = {"thermal": None}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_spatial_methods, dict)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_incompatible_options(self):
        options = {"bc_options": {"dimensionality": 1}}
        with self.assertRaises(pybamm.ModelError):
            pybamm.lead_acid.LOQS(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
