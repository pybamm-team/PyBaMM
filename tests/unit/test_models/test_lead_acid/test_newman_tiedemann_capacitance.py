#
# Tests for the lead-acid Newman-Tiedemann capacitance model
#
import pybamm
import unittest


class TestLeadAcidNewmanTiedemannCapacitance(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.NewmanTiedemannCapacitance()
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
