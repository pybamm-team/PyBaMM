#
# Tests for the lead-acid Newman-Tiedemann model
#
import pybamm
import unittest


class TestLeadAcidNewmanTiedemann(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"convection": {"transverse": "uniform"}}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()

        options = {"dimensionality": 1, "convection": {"transverse": "full"}}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_well_posed_1plus1d(self):
        options = {"dimensionality": 1}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()


class TestLeadAcidNewmanTiedemannSideReactions(unittest.TestCase):
    def test_well_posed(self):
        options = {"side reactions": ["oxygen"]}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()

    def test_well_posed_surface_form_differential(self):
        options = {"side reactions": ["oxygen"], "surface form": "differential"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_well_posed_surface_form_algebraic(self):
        options = {"side reactions": ["oxygen"], "surface form": "algebraic"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
