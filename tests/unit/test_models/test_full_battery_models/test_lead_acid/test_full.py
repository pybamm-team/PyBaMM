import pytest
import pybamm


@pytest.mark.parametrize(
    "options",
    [
        None,
        {"convection": "uniform transverse"},
        {"dimensionality": 1, "convection": "full transverse"},
    ],
    ids=[
        "well_posed",
        "well_posed_with_convection",
        "well_posed_with_convention_1plus1D",
    ],
)
def test_lead_acid_full(options):
    model = pybamm.lead_acid.Full(options)
    model.check_well_posedness()


@pytest.mark.parametrize(
    "options",
    [
        {"surface form": "differential"},
        {"surface form": "differential", "dimensionality": 1},
        {"surface form": "algebraic"},
    ],
    ids=[
        "well_posed_differential",
        "well_posed_differential_1plus1d",
        "well_posed_algebraic",
    ],
)
def test_lead_acid_full_surface_form(options):
    model = pybamm.lead_acid.Full(options)
    model.check_well_posedness()


@pytest.mark.parametrize(
    "options, expected_solver",
    [
        ({"hydrolysis": "true"}, None),
        ({"hydrolysis": "true", "surface form": "differential"}, pybamm.CasadiSolver),
        ({"hydrolysis": "true", "surface form": "algebraic"}, pybamm.CasadiSolver),
    ],
    ids=[
        "well_posed",
        "well_posed_surface_form_differential",
        "well_posed_surface_form_algebraic",
    ],
)
def test_lead_acid_full_side_reactions(options, expected_solver):
    model = pybamm.lead_acid.Full(options)
    model.check_well_posedness()
    if expected_solver:
        assert isinstance(model.default_solver, expected_solver)
