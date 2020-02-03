#
# Tests for the basic lithium-ion models
#
import pybamm
import unittest


class TestBasicModels(unittest.TestCase):
    def test_dfn_well_posed(self):
        model = pybamm.lithium_ion.BasicDFN()
        model.check_well_posedness()

    def test_dfn_default_geometry(self):
        model = pybamm.lithium_ion.BasicDFN()
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("secondary" in model.default_geometry["negative particle"])

    def test_spm_well_posed(self):
        model = pybamm.lithium_ion.BasicSPM()
        model.check_well_posedness()

    def test_spm_default_geometry(self):
        model = pybamm.lithium_ion.BasicSPM()
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("secondary" not in model.default_geometry["negative particle"])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
