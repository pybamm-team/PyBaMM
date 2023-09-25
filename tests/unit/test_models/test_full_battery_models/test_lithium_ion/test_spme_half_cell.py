#
# Tests for the lithium-ion half-cell SPMe model
# This is achieved by using the {"working electrode": "positive"} option
#
import pybamm
import unittest
from tests import TestCase
from tests import BaseUnitTestLithiumIonHalfCell


class TestSPMeHalfCell(BaseUnitTestLithiumIonHalfCell, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
