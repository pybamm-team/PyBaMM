import numpy as np
import pytest

import pybamm
import tests


class TestLOQS:
    def test_basic_processing(self):
        model = pybamm.lead_acid.LOQS()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lead_acid.LOQS()
        optimtest = tests.OptimisationsTest(model)
        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_allclose(original, to_python, rtol=1e-7, atol=1e-6)

    def test_set_up(self):
        model = pybamm.lead_acid.LOQS()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    @pytest.mark.parametrize("current_value", [-1, 0], ids=["charge", "zero_current"])
    def test_current(self, current_value):
        model = pybamm.lead_acid.LOQS()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Current function [A]": current_value})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
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
