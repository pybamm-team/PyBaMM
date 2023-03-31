#
# Tests for the basic lead acid models
#
from tests import TestCase
import pybamm
import unittest


class TestBasicModels(TestCase):
    def test_basic_full_lead_acid_well_posed(self):
        model = pybamm.lead_acid.BasicFull()
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
