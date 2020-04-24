#
# Tests for the lead-acid composite model
#
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidComposite(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.Composite()
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_basic_processing_with_convection(self):
        model = pybamm.lead_acid.Composite()
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lead_acid.Composite()
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

    def test_set_up(self):
        model = pybamm.lead_acid.Composite()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(simplify=False, to_python=True)
        optimtest.set_up_model(simplify=True, to_python=True)
        optimtest.set_up_model(simplify=False, to_python=False)
        optimtest.set_up_model(simplify=True, to_python=False)

    def test_basic_processing_1plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lead_acid.Composite(options)
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.y: 5,
            var.z: 5,
        }
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "convection": "full transverse",
        }
        model = pybamm.lead_acid.Composite(options)
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)


class TestLeadAcidCompositeSurfaceForm(unittest.TestCase):
    def test_basic_processing_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.Composite(options)
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_basic_processing_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.Composite(options)
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()  # solver=pybamm.CasadiSolver())

    # def test_thermal(self):
    #     options = {"thermal": "lumped"}
    #     model = pybamm.lead_acid.Composite(options)
    #     modeltest = tests.StandardModelTest(model)
    #     modeltest.test_all()

    #     options = {"thermal": "x-full"}
    #     model = pybamm.lead_acid.Composite(options)
    #     modeltest = tests.StandardModelTest(model)
    #     modeltest.test_all()


class TestLeadAcidCompositeExtended(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.CompositeExtended()
        param = model.default_parameter_values
        param.update({"Current function [A]": 1})
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_basic_processing_averaged(self):
        model = pybamm.lead_acid.CompositeAverageCorrection()
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
