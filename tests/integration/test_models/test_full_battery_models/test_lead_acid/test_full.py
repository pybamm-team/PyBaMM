import numpy as np
import pytest

import pybamm
import tests


@pytest.fixture
def optimtest():
    options = {"thermal": "isothermal"}
    model = pybamm.lead_acid.Full(options)
    return tests.OptimisationsTest(model)


class TestLeadAcidFull:
    @pytest.mark.parametrize(
        "options, t_eval",
        [
            ({"thermal": "isothermal"}, np.linspace(0, 3600 * 17)),
            (
                {"thermal": "isothermal", "convection": "uniform transverse"},
                np.linspace(0, 3600 * 10),
            ),
        ],
        ids=["basic_processing", "basic_processing_with_convection"],
    )
    def test_basic_processing(self, options, t_eval):
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=t_eval)

    def test_optimisations(self, optimtest):
        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_allclose(original, to_python, rtol=1e-7, atol=1e-6)

    def test_set_up(self, optimtest):
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

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
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)


class TestLeadAcidFullSurfaceForm:
    @pytest.mark.parametrize(
        "surface_form",
        ["differential", "algebraic"],
        ids=["differential", "algebraic"],
    )
    def test_basic_processing(self, surface_form):
        options = {"surface form": surface_form}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_set_up(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    @pytest.mark.parametrize(
        "options, param_update",
        [
            ({"thermal": "lumped"}, {"Current function [A]": 1.7}),
            ({"thermal": "x-full"}, None),
        ],
        ids=["lumped_with_current_function", "x_full"],
    )
    def test_thermal(self, options, param_update):
        model = pybamm.lead_acid.Full(options)
        param = model.default_parameter_values
        if param_update:
            param.update(param_update)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()
