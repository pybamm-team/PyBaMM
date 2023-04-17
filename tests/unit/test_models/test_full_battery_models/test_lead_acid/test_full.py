#
# Tests for the lead-acid Full model
#
from tests import TestCase
import pybamm
import unittest


class TestLeadAcidFull(TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.Full()
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"convection": "uniform transverse"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

        options = {"dimensionality": 1, "convection": "full transverse"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()


class TestLeadAcidFullSurfaceForm(TestCase):
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


class TestLeadAcidFullSideReactions(TestCase):
    def test_well_posed(self):
        options = {"hydrolysis": "true"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    def test_well_posed_surface_form_differential(self):
        options = {"hydrolysis": "true", "surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()
        self.assertIsInstance(model.default_solver, pybamm.CasadiSolver)

    def test_well_posed_surface_form_algebraic(self):
        options = {"hydrolysis": "true", "surface form": "algebraic"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()
        self.assertIsInstance(model.default_solver, pybamm.CasadiSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
