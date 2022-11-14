#
# Tests for current input functions
#
import pybamm
import numbers
import unittest
import numpy as np


class TestCurrentFunctions(unittest.TestCase):
    def test_constant_current(self):
        # test simplify
        param = pybamm.electrical_parameters
        current = param.current_with_time
        parameter_values = pybamm.ParameterValues({"Current function [A]": 2})
        processed_current = parameter_values.process_symbol(current)
        self.assertIsInstance(processed_current, pybamm.Scalar)

    def test_get_current_data(self):
        # test process parameters
        param = pybamm.electrical_parameters
        current = param.current_with_time
        parameter_values = pybamm.ParameterValues(
            {"Current function [A]": "[current data]car_current"}
        )
        current_eval = parameter_values.process_symbol(current)

        def current(t):
            return current_eval.evaluate(t=t)

        standard_tests = StandardCurrentFunctionTests([current], always_array=True)
        standard_tests.test_all()

    def test_user_current(self):
        # create user-defined sin function
        def my_fun(t, A, omega):
            return A * pybamm.sin(2 * np.pi * omega * t)

        # choose amplitude and frequency
        param = pybamm.electrical_parameters
        A = 5
        omega = pybamm.Parameter("omega")

        def current(t):
            return my_fun(t, A, omega)

        # set and process parameters
        parameter_values = pybamm.ParameterValues(
            {
                "omega": 3,
                "Current function [A]": current,
            }
        )
        current = param.current_with_time
        current_eval = parameter_values.process_symbol(current)

        def user_current(t):
            return current_eval.evaluate(t=t)

        # check output types
        standard_tests = StandardCurrentFunctionTests([user_current])
        standard_tests.test_all()

        # check output correct value
        time = np.linspace(0, 3600, 600)
        np.testing.assert_array_almost_equal(
            user_current(time), 5 * np.sin(2 * np.pi * 3 * time)
        )


class StandardCurrentFunctionTests(object):
    def __init__(self, function_list, always_array=False):
        self.function_list = function_list
        self.always_array = always_array

    def test_output_type(self):
        for function in self.function_list:
            if self.always_array is True:
                assert isinstance(function(0), np.ndarray)
            else:
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
    pybamm.settings.debug_mode = True
    unittest.main()
