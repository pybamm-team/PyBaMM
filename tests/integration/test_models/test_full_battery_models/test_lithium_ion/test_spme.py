#
# Tests for the lithium-ion SPMe model
#
from __future__ import annotations

import unittest

import pybamm
from tests import BaseIntegrationTestLithiumIon, TestCase


class TestSPMe(BaseIntegrationTestLithiumIon, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.run_basic_processing_test(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
