#
# Tests for the lead-acid LOQS model
#
from tests import TestCase
import pybamm
import tests

import unittest
import numpy as np


class TestLOQS(TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.LOQS()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lead_acid.LOQS()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, to_python)

    def test_set_up(self):
        model = pybamm.lead_acid.LOQS()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    def test_charge(self):
        model = pybamm.lead_acid.LOQS()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Current function [A]": -1})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_zero_current(self):
        model = pybamm.lead_acid.LOQS()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Current function [A]": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_basic_processing_with_convection(self):
        model = pybamm.lead_acid.LOQS({"convection": "uniform transverse"})
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_thermal(self):
        options = {"thermal": "lumped"}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

        options = {"thermal": "x-full"}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_basic_processing_1plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lead_acid.LOQS(options)
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "y": 5, "z": 5}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "convection": "full transverse",
        }
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
