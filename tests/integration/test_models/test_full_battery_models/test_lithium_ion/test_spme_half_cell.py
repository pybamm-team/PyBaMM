#
# Tests for the half-cell lithium-ion SPMe model
#
from __future__ import annotations

import unittest

import pybamm
from tests import BaseIntegrationTestLithiumIonHalfCell, TestCase


class TestSPMeHalfCell(BaseIntegrationTestLithiumIonHalfCell, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
