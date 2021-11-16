#
# Tests for the lithium-ion DFN model
#
import pybamm
import unittest
from tests import TestLithiumIonModel


class TestDFN(TestLithiumIonModel):
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.lithium_ion.DFN(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
