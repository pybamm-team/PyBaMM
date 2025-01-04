import pybamm
import tests
import pytest
import numpy as np


@pytest.fixture(
    params=[
        {"surface form": "algebraic"},
        {"surface form": "differential"},
        {
            "surface form": "algebraic",
            "current collector": "potential pair",
            "dimensionality": 1,
        },
    ],
    ids=[
        "basic_processing_algebraic",
        "basic_processing_with_capacitance",
        "basic_processing_1p1D_algebraic",
    ],
)
def loqs_surface_model(request):
    return pybamm.lead_acid.LOQS(request.param), request.param


def test_basic_processing(loqs_surface_model):
    model, options = loqs_surface_model
    model_test = tests.StandardModelTest(model)
    skip_output_tests = options.get("dimensionality") == 1
    model_test.test_all(skip_output_tests=skip_output_tests)


def test_optimisations():
    options = {"surface form": "differential"}
    model = pybamm.lead_acid.LOQS(options)
    optimtest = tests.OptimisationsTest(model)

    original = optimtest.evaluate_model()
    to_python = optimtest.evaluate_model(to_python=True)
    np.testing.assert_array_almost_equal(original, to_python, decimal=5)


def test_set_up():
    options = {"surface form": "differential"}
    model = pybamm.lead_acid.LOQS(options)
    optimtest = tests.OptimisationsTest(model)
    optimtest.set_up_model(to_python=True)
    optimtest.set_up_model(to_python=False)


@pytest.mark.skip(reason="model not working for 1+1D differential")
def test_basic_processing_1p1D_differential():
    options = {
        "surface form": "differential",
        "current collector": "potential pair",
        "dimensionality": 1,
    }
    model = pybamm.lead_acid.LOQS(options)
    model_test = tests.StandardModelTest(model)
    model_test.test_all(skip_output_tests=True)
