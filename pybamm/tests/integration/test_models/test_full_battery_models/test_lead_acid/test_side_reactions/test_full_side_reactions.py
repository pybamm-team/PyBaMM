#
# Tests for the lead-acid Full model
#
from tests import TestCase
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidFullSideReactions(TestCase):
    def test_basic_processing(self):
        options = {"hydrolysis": "true"}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True, t_eval=np.linspace(0, 3600 * 17))

    def test_basic_processing_differential(self):
        options = {"hydrolysis": "true", "surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_algebraic(self):
        options = {"hydrolysis": "true", "surface form": "algebraic"}
        model = pybamm.lead_acid.Full(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_charge(self):
        options = {"hydrolysis": "true", "surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Current function [A]": -1, "Initial State of Charge": 0.5}
        )
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_zero_current(self):
        options = {"hydrolysis": "true", "surface form": "differential"}
        model = pybamm.lead_acid.Full(options)
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
