#
# Tests for the lead-acid composite model
#
import pybamm
import unittest


class TestLeadAcidComposite(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None, "Voltage": "On"}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"thermal": None, "Voltage": "On", "convection": True}
        model = pybamm.lead_acid.Composite(options)
        model.check_well_posedness()


class TestLeadAcidCompositeSurfaceForm(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None, "Voltage": "On", "capacitance": False}
        model = pybamm.lead_acid.surface_form.Composite(options)
        model.check_well_posedness()

    def test_well_posed_with_capacitance(self):
        options = {"thermal": None, "Voltage": "On", "capacitance": True}
        model = pybamm.lead_acid.surface_form.Composite(options)
        model.check_well_posedness()

    def test_default_solver(self):
        options = {"thermal": None, "Voltage": "On", "capacitance": True}
        model = pybamm.lead_acid.surface_form.Composite(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsOdeSolver)
        options = {"thermal": None, "Voltage": "On", "capacitance": False}
        model = pybamm.lead_acid.surface_form.Composite(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_well_posed_average_first_order(self):
        model = pybamm.lead_acid.Composite({"first-order potential": "average"})
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        model = pybamm.lead_acid.Composite({"convection": True})
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
