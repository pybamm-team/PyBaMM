#
# Tests for the lithium-ion half-cell DFN model
#
from tests import TestCase
import pybamm
import unittest
from tests import BaseUnitTestLithiumIonHalfCell


class TestDFNHalfCell(BaseUnitTestLithiumIonHalfCell, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
