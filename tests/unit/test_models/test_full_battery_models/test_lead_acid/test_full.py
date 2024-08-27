#
# Tests for the lead-acid Full model
#
import pybamm


class TestLeadAcidFull:
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


class TestLeadAcidFullSurfaceForm:
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


class TestLeadAcidFullSideReactions:
    def test_well_posed(self):
        options = {"hydrolysis": "true"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    def test_well_posed_surface_form_differential(self):
        options = {"hydrolysis": "true", "surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()
        assert isinstance(model.default_solver, pybamm.CasadiSolver)

    def test_well_posed_surface_form_algebraic(self):
        options = {"hydrolysis": "true", "surface form": "algebraic"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()
        assert isinstance(model.default_solver, pybamm.CasadiSolver)
