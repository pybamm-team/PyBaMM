#
# Tests for the lead-acid composite model
#
import pybamm
import unittest


class TestLeadAcidComposite(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.Composite()
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"convection": True}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()


class TestLeadAcidCompositeSurfaceForm(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    def test_well_posed_differential(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"convection": True}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
