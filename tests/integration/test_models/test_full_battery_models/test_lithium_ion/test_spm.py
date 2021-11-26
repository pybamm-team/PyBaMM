#
# Tests for the lithium-ion SPM model
#
import pybamm
import unittest
from tests import BaseIntegrationTestLithiumIon


class TestSPM(BaseIntegrationTestLithiumIon, unittest.TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    sys.setrecursionlimit(10000)

    if "-v" in sys.argv:
        debug = True
    unittest.main()
