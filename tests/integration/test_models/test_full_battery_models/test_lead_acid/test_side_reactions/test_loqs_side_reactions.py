#
# Tests for the lead-acid LOQS model
#
from tests import TestCase
import pybamm
import tests
import unittest


class TestLeadAcidLOQSWithSideReactions(TestCase):
    def test_discharge_differential(self):
        options = {"surface form": "differential", "hydrolysis": "true"}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_discharge_differential_varying_surface_area(self):
        options = {
            "surface form": "differential",
            "hydrolysis": "true",
        }
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_discharge_algebraic(self):
        options = {"surface form": "algebraic", "hydrolysis": "true"}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    def test_charge(self):
        options = {"surface form": "differential", "hydrolysis": "true"}
        model = pybamm.lead_acid.LOQS(options)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Current function [A]": -1, "Initial State of Charge": 0.5}
        )
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)

    def test_zero_current(self):
        options = {"surface form": "differential", "hydrolysis": "true"}
        model = pybamm.lead_acid.LOQS(options)
        parameter_values = model.default_parameter_values
        parameter_values.update({"Current function [A]": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
