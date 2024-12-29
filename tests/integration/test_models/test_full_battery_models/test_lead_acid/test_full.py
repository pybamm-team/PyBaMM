import pybamm
import pytest


@pytest.fixture(
    params=[
        (None, "well_posed"),
        ({"convection": "uniform transverse"}, "well_posed"),
        ({"dimensionality": 1, "convection": "full transverse"}, "well_posed"),
        ({"surface form": "differential"}, "well_posed_differential"),
        (
            {"surface form": "differential", "dimensionality": 1},
            "well_posed_differential_1plus1d",
        ),
        ({"surface form": "algebraic"}, "well_posed_algebraic"),
        ({"hydrolysis": "true"}, "well_posed_hydrolysis"),
        (
            {"hydrolysis": "true", "surface form": "differential"},
            "well_posed_surface_form_differential",
        ),
        (
            {"hydrolysis": "true", "surface form": "algebraic"},
            "well_posed_surface_form_algebraic",
        ),
    ]
)
def lead_acid_test_cases(request):
    options, test_name = request.param
    model = pybamm.lead_acid.Full(options)
    return model, test_name


def test_lead_acid_full(lead_acid_test_cases):
    model, test_name = lead_acid_test_cases

    model.check_well_posedness()

    if test_name in [
        "well_posed_surface_form_differential",
        "well_posed_surface_form_algebraic",
    ]:
        assert isinstance(model.default_solver, pybamm.CasadiSolver)
