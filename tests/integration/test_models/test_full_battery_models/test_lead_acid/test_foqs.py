#
# Tests for the lead-acid FOQS model
#
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidFOQS(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.FOQS()
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_basic_processing_with_convection(self):
        options = {"convection": "uniform transverse"}
        model = pybamm.lead_acid.FOQS(options)
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lead_acid.FOQS()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, to_python)

    def test_set_up(self):
        model = pybamm.lead_acid.FOQS()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)


class TestLeadAcidFOQSSurfaceForm(unittest.TestCase):
    def test_basic_processing_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.FOQS(options)
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_basic_processing_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.FOQS(options)
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
