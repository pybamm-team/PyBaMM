#
# Tests for the lead-acid Full model
#
import pybamm
import unittest


class TestLeadAcidFull(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.Full()
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"thermal": "isothermal", "convection": True}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    @unittest.skipIf(~pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        model = pybamm.lead_acid.Full()
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


class TestLeadAcidFullSurfaceForm(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    def test_well_posed_differential_1plus1d(self):
        options = {"surface form": "differential", "dimensionality": 1}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    @unittest.skipIf(~pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.Full(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


class TestLeadAcidFullSideReactions(unittest.TestCase):
    def test_well_posed(self):
        options = {"side reactions": ["oxygen"]}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    def test_well_posed_surface_form_differential(self):
        options = {"side reactions": ["oxygen"], "surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    def test_well_posed_surface_form_algebraic(self):
        options = {"side reactions": ["oxygen"], "surface form": "algebraic"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
