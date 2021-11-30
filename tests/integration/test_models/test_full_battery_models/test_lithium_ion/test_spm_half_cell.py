#
# Tests for the half-cell lithium-ion SPM model
#
import pybamm
import unittest
from tests import BaseIntegrationTestLithiumIonHalfCell


class TestSPMHalfCell(BaseIntegrationTestLithiumIonHalfCell, unittest.TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
