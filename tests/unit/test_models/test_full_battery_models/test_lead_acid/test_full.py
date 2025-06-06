import pytest

import pybamm


class TestLeadAcidFull:
    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"convection": "uniform transverse"},
            {"dimensionality": 1, "convection": "full transverse"},
        ],
        ids=["well_posed", "with_convention", "with_convention_1plus1d"],
    )
    def test_well_posed(self, options):
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()


class TestLeadAcidFullSurfaceForm:
    @pytest.mark.parametrize(
        "options",
        [
            {"surface form": "differential"},
            {"surface form": "differential", "dimensionality": 1},
            {"surface form": "algebraic"},
        ],
        ids=["differential", "differential_1plus1d", "algebraic"],
    )
    def test_well_posed(self, options):
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()


class TestLeadAcidFullSideReactions:
    def test_model_well_posedness(self):
        options = {"hydrolysis": "true"}
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()

    @pytest.mark.parametrize(
        "options, expected_solver",
        [
            (
                {"hydrolysis": "true", "surface form": "differential"},
                pybamm.IDAKLUSolver,
            ),
            ({"hydrolysis": "true", "surface form": "algebraic"}, pybamm.IDAKLUSolver),
        ],
        ids=[
            "surface_form_differential",
            "surface_form_algebraic",
        ],
    )
    def test_well_posed(self, options, expected_solver):
        model = pybamm.lead_acid.Full(options)
        model.check_well_posedness()
        if expected_solver is not None:
            assert isinstance(model.default_solver, expected_solver)
