#
# Tests for the lead-acid LOQS model with capacitance
#
from tests import TestCase
import pybamm
import tests

import unittest

import numpy as np


class TestLeadAcidLoqsSurfaceForm(TestCase):
    def test_basic_processing(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_basic_processing_with_capacitance(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @unittest.skip("model not working for 1+1D differential")
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

    def test_optimisations(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, to_python, decimal=5)

    def test_set_up(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
