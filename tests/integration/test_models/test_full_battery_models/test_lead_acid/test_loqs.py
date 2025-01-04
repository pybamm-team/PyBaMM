import pytest
import pybamm
import numpy as np
import tests


@pytest.fixture
def loqs_model():
    return pybamm.lead_acid.LOQS()


@pytest.fixture(
    params=[
        {},
        {"convection": "uniform transverse"},
        {"thermal": "lumped"},
        {"thermal": "x-full"},
        {"current collector": "potential pair", "dimensionality": 1},
        {
            "current collector": "potential pair",
            "dimensionality": 1,
            "convection": "full transverse",
        },
    ],
    ids=[
        "basic_processing",
        "basic_processing_with_convection",
        "thermal_lumped",
        "thermal_x_full",
        "basic_processing_1plus1D",
        "basic_processing_1plus1D_with_convection",
    ],
)
def loqs_model_with_options(request):
    return pybamm.lead_acid.LOQS(request.param), request.param


def test_basic_processing(loqs_model):
    modeltest = tests.StandardModelTest(loqs_model)
    modeltest.test_all()


def test_optimisations(loqs_model):
    optimtest = tests.OptimisationsTest(loqs_model)
    original = optimtest.evaluate_model()
    to_python = optimtest.evaluate_model(to_python=True)
    np.testing.assert_array_almost_equal(original, to_python)


@pytest.mark.parametrize("current", [-1, 0], ids=["charge", "zero_current"])
def test_current_dependency(loqs_model, current):
    parameter_values = loqs_model.default_parameter_values
    parameter_values.update({"Current function [A]": current})
    modeltest = tests.StandardModelTest(loqs_model, parameter_values=parameter_values)
    modeltest.test_all()


def test_model_with_options(loqs_model_with_options):
    model, options = loqs_model_with_options
    var_pts = None
    if options.get("dimensionality") == 1:
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "y": 5, "z": 5}
    modeltest = tests.StandardModelTest(model, var_pts=var_pts)
    skip_output_tests = options.get("dimensionality") == 1
    modeltest.test_all(skip_output_tests=skip_output_tests)
