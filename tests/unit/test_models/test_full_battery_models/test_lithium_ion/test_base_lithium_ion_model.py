#
# Tests for the base lead acid model class
#
import pybamm
import unittest


class TestBaseLithiumIonModel(unittest.TestCase):
    def test_incompatible_options(self):
        with self.assertRaisesRegex(pybamm.OptionError, "convection not implemented"):
            pybamm.lithium_ion.BaseModel({"convection": "uniform transverse"})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
