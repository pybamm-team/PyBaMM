#
# Tests for the lithium-ion DFN model
#
from __future__ import annotations

import unittest

import pybamm
from tests import TestCase


class TestYang2017(TestCase):
    def test_well_posed(self):
        model = pybamm.lithium_ion.Yang2017()
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
