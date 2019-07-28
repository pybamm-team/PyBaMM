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
        options = {"convection": True}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    def test_well_posed_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lead_acid.Composite(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.Composite(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


class TestLeadAcidCompositeWithSideReactions(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    def test_varying_surface_area(self):
        options = {
            "surface form": "differential",
            "side reactions": ["oxygen"],
            "interfacial surface area": "varying",
        }
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

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
