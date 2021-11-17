#
# Tests for the lithium-ion Newman-Tobias model
#
import pybamm
import unittest
from tests import BaseIntegrationTestLithiumIon


class TestNewmanTobias(BaseIntegrationTestLithiumIon, unittest.TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.NewmanTobias

    def test_sensitivities(self):
        pass  # skip this test


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
