#
# Tests for the lead-acid composite model
#
import pybamm
import unittest


class TestLeadAcidComposite(unittest.TestCase):
    def test_well_posed(self):
        # debug mode slows down the composite model a fair bit, so turn off
        pybamm.setting.debug_mode = False
        model = pybamm.lead_acid.Composite()
        pybamm.setting.debug_mode = True
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"convection": True}
        # debug mode slows down the composite model a fair bit, so turn off
        pybamm.setting.debug_mode = False
        model = pybamm.lead_acid.Composite(options)
        pybamm.setting.debug_mode = True
        model.check_well_posedness()

    def test_well_posed_differential(self):
        options = {"surface form": "differential"}
        # debug mode slows down the composite model a fair bit, so turn off
        pybamm.setting.debug_mode = False
        model = pybamm.lead_acid.Composite(options)
        pybamm.setting.debug_mode = True
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
