#
# Tests for FOQS lead-acid model
#
import pybamm
import unittest


class TestLeadAcidFOQS(unittest.TestCase):
    def test_well_posed(self):
        # debug mode slows down the FOQS model a fair bit, so turn off
        pybamm.settings.debug_mode = False
        model = pybamm.lead_acid.FOQS()
        pybamm.settings.debug_mode = True
        model.check_well_posedness()


class TestLeadAcidFOQSWithSideReactions(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential", "side reactions": ["oxygen"]}
        # debug mode slows down the FOQS model a fair bit, so turn off
        pybamm.settings.debug_mode = False
        model = pybamm.lead_acid.FOQS(options)
        pybamm.settings.debug_mode = True
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
