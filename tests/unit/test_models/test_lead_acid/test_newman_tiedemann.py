#
# Tests for the lead-acid Newman-Tiedemann model
#
import pybamm
import unittest


class TestLeadAcidNewmanTiedemann(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        model.check_well_posedness()

    def test_default_solver(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


class TestLeadAcidNewmanTiedemannCapacitance(unittest.TestCase):
    def test_well_posed(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()

    def test_well_posed_no_capacitance(self):
        options = {"capacitance": "algebraic"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        model.check_well_posedness()

    def test_default_solver(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsOdeSolver)
        options = {"capacitance": "algebraic"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
