import pybamm
import tests
import pytest
import numpy as np


@pytest.fixture
def loqs_model():
    return pybamm.lead_acid.LOQS()


class TestLOQS:
    def test_basic_processing(self, loqs_model):
        modeltest = tests.StandardModelTest(loqs_model)
        modeltest.test_all()

    def test_optimisations(self, loqs_model):
        optimtest = tests.OptimisationsTest(loqs_model)
        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, to_python)

    def test_set_up(self, loqs_model):
        optimtest = tests.OptimisationsTest(loqs_model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    @pytest.mark.parametrize("current_value", [-1, 0], ids=["charge", "zero_current"])
    def test_current(self, current_value, loqs_model):
        parameter_values = loqs_model.default_parameter_values
        parameter_values.update({"Current function [A]": current_value})
        modeltest = tests.StandardModelTest(
            loqs_model, parameter_values=parameter_values
        )
        modeltest.test_all()

    def test_basic_processing_with_convection(self):
        model = pybamm.lead_acid.LOQS({"convection": "uniform transverse"})
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @pytest.mark.parametrize(
        "options",
        [{"thermal": "lumped"}, {"thermal": "x-full"}],
        ids=["lumped", "x_full"],
    )
    def test_thermal(self, options):
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @pytest.mark.parametrize(
        "options",
        [
            {"current collector": "potential pair", "dimensionality": 1},
            {
                "current collector": "potential pair",
                "dimensionality": 1,
                "convection": "full transverse",
            },
        ],
        ids=["basic_1plus1D", "1plus1D_with_convection"],
    )
    def test_basic_processing_1plus1D(self, options):
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "y": 5, "z": 5}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)
