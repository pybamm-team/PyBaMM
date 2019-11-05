#
# Tests for the lead-acid Full model
#
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidCompositeSideReactions(unittest.TestCase):
    def test_basic_processing_differential(self):
        options = {"side reactions": ["oxygen"], "surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_basic_processing_algebraic(self):
        options = {"side reactions": ["oxygen"], "surface form": "algebraic"}
        model = pybamm.lead_acid.Composite(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_charge(self):
        options = {"side reactions": ["oxygen"], "surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Typical current [A]": -1, "Initial State of Charge": 0.5}
        )
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_zero_current(self):
        options = {"side reactions": ["oxygen"], "surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Current function": pybamm.ConstantCurrent(current=0)}
        )
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_extended_differential(self):
        options = {"side reactions": ["oxygen"], "surface form": "differential"}
        model = pybamm.lead_acid.CompositeExtended(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self):
        options = {"side reactions": ["oxygen"], "surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified, decimal=5)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known, decimal=5)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
