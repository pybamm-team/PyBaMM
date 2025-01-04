import pybamm
import tests
import pytest
import numpy as np


@pytest.fixture(
    params=[
        {"thermal": "isothermal"},
        {"thermal": "isothermal", "convection": "uniform transverse"},
    ],
    ids=[
        "isothermal",
        "isothermal_with_convection",
    ],
)
def full_model_options(request):
    return request.param


@pytest.fixture(
    params=[
        {"surface form": "differential"},
        {"surface form": "algebraic"},
        {"thermal": "lumped"},
        {"thermal": "x-full"},
    ],
    ids=[
        "surface_differential",
        "surface_algebraic",
        "thermal_lumped",
        "thermal_x_full",
    ],
)
def full_surface_model_options(request):
    return request.param


def test_basic_processing(full_model_options):
    model = pybamm.lead_acid.Full(full_model_options)
    modeltest = tests.StandardModelTest(model)
    t_eval = (
        np.linspace(0, 3600 * 10)
        if "convection" in full_model_options
        else np.linspace(0, 3600 * 17)
    )
    skip_output_tests = full_model_options.get("dimensionality") == 1
    modeltest.test_all(t_eval=t_eval, skip_output_tests=skip_output_tests)


@pytest.fixture(
    params=[
        {"current collector": "potential pair", "dimensionality": 1},
        {
            "current collector": "potential pair",
            "dimensionality": 1,
            "convection": "full transverse",
        },
    ],
    ids=["basic", "with_convection"],
)
def model_1plus1D_options(request):
    return request.param


def test_basic_processing_1plus1D(model_1plus1D_options):
    var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "y": 5, "z": 5}
    model = pybamm.lead_acid.Full(model_1plus1D_options)
    modeltest = tests.StandardModelTest(model, var_pts=var_pts)
    modeltest.test_all(skip_output_tests=True)


def test_optimisations():
    options = {"thermal": "isothermal"}
    model = pybamm.lead_acid.Full(options)
    optimtest = tests.OptimisationsTest(model)
    original = optimtest.evaluate_model()
    to_python = optimtest.evaluate_model(to_python=True)
    np.testing.assert_array_almost_equal(original, to_python)


@pytest.fixture(
    params=[
        {"thermal": "isothermal"},
        {"surface form": "differential"},
    ],
    ids=["full", "surface_form"],
)
def setup_options(request):
    return request.param


def test_set_up(setup_options):
    model = pybamm.lead_acid.Full(setup_options)
    optimtest = tests.OptimisationsTest(model)
    optimtest.set_up_model(to_python=True)
    optimtest.set_up_model(to_python=False)


def test_surface_processing(full_surface_model_options):
    model = pybamm.lead_acid.Full(full_surface_model_options)
    modeltest = tests.StandardModelTest(model)
    if full_surface_model_options.get("thermal") == "lumped":
        param = model.default_parameter_values
        param["Current function [A]"] = 1.7
        modeltest = tests.StandardModelTest(model, parameter_values=param)
    modeltest.test_all()
