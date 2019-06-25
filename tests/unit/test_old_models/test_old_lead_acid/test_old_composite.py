#
# Tests for the lead-acid composite model
#
import pybamm
import unittest


class TestOldLeadAcidComposite(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.old_lead_acid.OldComposite()
        model.check_well_posedness()

    def test_well_posed_average_first_order(self):
        model = pybamm.old_lead_acid.OldComposite({"first-order potential": "average"})
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        model = pybamm.old_lead_acid.OldComposite({"convection": True})
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
unittest.main()
