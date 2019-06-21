#
# Tests for the lithium-ion SPM model
#
import pybamm
import tests
import numpy as np
import unittest


class TestSPM(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.SPM()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_basic_processing_2plus1D(self):
        options = {"bc_options": {"dimensionality": 2}}
        model = pybamm.lithium_ion.SPM(options)
        modeltest = tests.StandardModelTest(model)
        # TO DO: fix processed variable for 3D variables which come from outer
        # product with current collector variables
        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self):
        model = pybamm.lithium_ion.SPM()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        simp_and_python = optimtest.evaluate_model(simplify=True, to_python=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)
        np.testing.assert_array_almost_equal(original, simp_and_python)

    def test_charge(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current [A]": -1})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_zero_current(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current [A]": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
