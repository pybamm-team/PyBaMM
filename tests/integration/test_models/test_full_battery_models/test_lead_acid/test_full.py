#
# Tests for the lead-acid Full model
#
import pybamm
import tests

import numpy as np


class TestLeadAcidFull:
    def test_basic_processing(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 3600 * 17))

    def test_basic_processing_with_convection(self):
        options = {"thermal": "isothermal", "convection": "uniform transverse"}
        model = pybamm.lead_acid.Full(options)
        # var_pts = {"x_n": 10, "x_s": 10, "x_p": 10}
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 3600 * 10))

    def test_optimisations(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lead_acid.Full(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, to_python)

    def test_set_up(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lead_acid.Full(options)
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    def test_basic_processing_1plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lead_acid.Full(options)
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "y": 5, "z": 5}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "convection": "full transverse",
        }
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)


class TestLeadAcidFullSurfaceForm:
    def test_basic_processing_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_basic_processing_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_set_up(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    def test_thermal(self):
        options = {"thermal": "lumped"}
        model = pybamm.lead_acid.Full(options)
        param = model.default_parameter_values
        param["Current function [A]"] = 1.7
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

        options = {"thermal": "x-full"}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
