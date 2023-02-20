#
# Tests for the lithium-ion SPMe model
#
import pybamm
import unittest
from tests import BaseUnitTestLithiumIon


class TestSPMe(BaseUnitTestLithiumIon, unittest.TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.lithium_ion.SPMe(options)

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.check_well_posedness(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
