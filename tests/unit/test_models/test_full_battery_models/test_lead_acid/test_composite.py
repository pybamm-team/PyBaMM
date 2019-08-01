#
# Tests for the lead-acid composite models
#
import pybamm
import unittest


class TestLeadAcidComposite(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.Composite()
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        # this test is very slow with debug mode set to true
        pybamm.settings.debug_mode = False
        options = {"convection": {"transverse": "uniform"}}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

        options = {"dimensionality": 1, "convection": {"transverse": "full"}}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()
        pybamm.settings.debug_mode = True

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_well_posed_1plus1D(self):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


class TestLeadAcidCompositeWithSideReactions(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_incompatible_options(self):
        options = {"side reactions": ["something"]}
        with self.assertRaises(pybamm.OptionError):
            pybamm.lead_acid.Composite(options)


class TestLeadAcidCompositeExtended(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.CompositeExtended()
        model.check_well_posedness()

    def test_well_posed_differential_side_reactions(self):
        options = {"surface form": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.CompositeExtended(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
