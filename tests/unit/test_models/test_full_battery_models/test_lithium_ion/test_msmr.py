#
# Tests for the lithium-ion MSMR model
#
from __future__ import annotations

import unittest

import pybamm
from tests import TestCase


class TestMSMR(TestCase):
    def test_well_posed(self):
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
