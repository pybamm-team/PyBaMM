#
# Tests for the lithium-ion SPM model
#
from __future__ import annotations

import unittest

import pybamm
from tests import BaseIntegrationTestLithiumIon, TestCase


class TestSPM(BaseIntegrationTestLithiumIon, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
