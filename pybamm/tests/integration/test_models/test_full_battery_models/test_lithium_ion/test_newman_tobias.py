#
# Tests for the lithium-ion Newman-Tobias model
#
from tests import TestCase
import pybamm
import unittest
from tests import BaseIntegrationTestLithiumIon


class TestNewmanTobias(BaseIntegrationTestLithiumIon, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.NewmanTobias

    def test_basic_processing(self):
        options = {}
        self.run_basic_processing_test(options)

    def test_sensitivities(self):
        pass  # skip this test

    def test_charge(self):
        pass  # skip this test

    def test_composite_graphite_silicon(self):
        pass  # skip this test

    def test_composite_graphite_silicon_sei(self):
        pass  # skip this test


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
