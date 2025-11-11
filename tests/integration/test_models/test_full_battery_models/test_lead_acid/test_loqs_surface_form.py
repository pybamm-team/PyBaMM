import numpy as np
import pytest

import pybamm
import tests


@pytest.fixture
def optimtest():
    options = {"surface form": "differential"}
    model = pybamm.lead_acid.LOQS(options)
    optimtest_instance = tests.OptimisationsTest(model)
    return optimtest_instance


class TestLeadAcidLoqsSurfaceForm:
    @pytest.mark.parametrize(
        "surface_form",
        ["algebraic", "differential"],
        ids=["basic_processing", "basic_processing_with_capacitance"],
    )
    def test_basic_processing(self, surface_form):
        options = {"surface form": surface_form}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @pytest.mark.skip(reason="model not working for 1+1D differential")
    def test_basic_processing_1p1D_differential(self):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_1p1D_algebraic(self):
        options = {
            "surface form": "algebraic",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self, optimtest):
        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_allclose(original, to_python, rtol=1e-6, atol=1e-5)

    def test_set_up(self, optimtest):
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)
