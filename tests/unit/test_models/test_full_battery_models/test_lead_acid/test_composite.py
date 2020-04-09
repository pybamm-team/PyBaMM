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
        options = {"convection": "uniform transverse"}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

        options = {"dimensionality": 1, "convection": "full transverse"}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()
        pybamm.settings.debug_mode = True


class TestLeadAcidCompositeMultiDimensional(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.Composite(
            {"dimensionality": 1, "current collector": "potential pair"}
        )
        self.assertIsInstance(
            model.default_solver, (pybamm.ScikitsDaeSolver, pybamm.CasadiSolver)
        )
        model.check_well_posedness()

        model = pybamm.lead_acid.Composite(
            {"dimensionality": 2, "current collector": "potential pair"}
        )
        model.check_well_posedness()

        model = pybamm.lead_acid.Composite(
            {
                "dimensionality": 1,
                "current collector": "potential pair quite conductive",
            }
        )
        model.check_well_posedness()

        model = pybamm.lead_acid.Composite(
            {
                "dimensionality": 2,
                "current collector": "potential pair quite conductive",
            }
        )
        model.check_well_posedness()


class TestLeadAcidCompositeWithSideReactions(unittest.TestCase):
    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()
        self.assertIsInstance(
            model.default_solver, (pybamm.ScikitsDaeSolver, pybamm.CasadiSolver)
        )


class TestLeadAcidCompositeExtended(unittest.TestCase):
    def test_well_posed_differential_side_reactions(self):
        options = {"surface form": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.CompositeExtended(options)
        model.check_well_posedness()

    def test_well_posed_average_correction(self):
        model = pybamm.lead_acid.CompositeAverageCorrection()
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
