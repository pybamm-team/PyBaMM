#
# Tests for the lithium-ion half-cell SPM model
#
from __future__ import annotations

import unittest

import pybamm
from tests import BaseUnitTestLithiumIonHalfCell, TestCase


class TestSPMHalfCell(BaseUnitTestLithiumIonHalfCell, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
