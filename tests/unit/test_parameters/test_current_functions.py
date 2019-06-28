#
# Tests for current input functions
#
import pybamm
import pybamm.parameters.standard_current_functions as cf
import numbers
import os
import unittest
import numpy as np


class TestCurrentFunctions(unittest.TestCase):
    def test_all_functions(self):
        function_list = [cf.sin_current, cf.car_current, cf.get_csv_current]
        standard_tests = StandardCurrentFunctionTests(function_list)
        standard_tests.test_all()

    def test_sin_current(self):
        # create current functions
        current_density = (
            pybamm.electrical_parameters.dimensional_current_density_with_time
        )

        # process
        tau = 1800
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height [m]": 0.1,
                "Electrode depth [m]": 0.1,
                "Typical timescale [s]": tau,
                "Number of electrodes connected in parallel to make a cell": 8,
                "Typical current [A]": 2,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "sin_current.py",
                ),
            }
        )
        current_desnity_eval = parameter_values.process_symbol(current_density)
        # one hour dimensional time
        time = np.linspace(0, 3600, 600)
        np.testing.assert_array_almost_equal(
            current_desnity_eval.evaluate(t=time / tau),
            (2 / (8 * 0.1 * 0.1)) * np.sin(2 * np.pi * time),
        )


class StandardCurrentFunctionTests(object):
    def __init__(self, function_list):
        self.function_list = function_list

    def test_output_type(self):
        for function in self.function_list:
            assert isinstance(function(0), numbers.Number)
            assert isinstance(function(np.zeros(3)), np.ndarray)
            assert isinstance(function(np.zeros([3, 3])), np.ndarray)

    def test_all(self):
        self.test_output_type()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
