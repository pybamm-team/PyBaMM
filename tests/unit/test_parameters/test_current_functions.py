#
# Tests for current input functions
#
import pybamm
import numbers
import unittest
import numpy as np


class TestCurrentFunctions(unittest.TestCase):
    def test_constant_current(self):
        function = pybamm.GetConstantCurrent(current=4)
        assert isinstance(function(0), numbers.Number)
        assert isinstance(function(np.zeros(3)), numbers.Number)
        assert isinstance(function(np.zeros([3, 3])), numbers.Number)

        # test simplify
        current = pybamm.electrical_parameters.current_with_time
        parameter_values = pybamm.ParameterValues(
            {
                "Typical current [A]": 2,
                "Typical timescale [s]": 1,
                "Current function": pybamm.GetConstantCurrent(),
            }
        )
        processed_current = parameter_values.process_symbol(current)
        self.assertIsInstance(processed_current.simplify(), pybamm.Scalar)

    def test_get_current_data(self):
        # test units
        function_list = [
            pybamm.GetCurrentData("US06.csv", units="[A]"),
            pybamm.GetCurrentData("car_current.csv", units="[]", current_scale=10),
        ]
        for function in function_list:
            function.interpolate()

        # test process parameters
        dimensional_current = pybamm.electrical_parameters.dimensional_current_with_time
        parameter_values = pybamm.ParameterValues(
            {
                "Typical current [A]": 2,
                "Typical timescale [s]": 1,
                "Current function": pybamm.GetCurrentData(
                    "car_current.csv", units="[]"
                ),
            }
        )
        dimensional_current_eval = parameter_values.process_symbol(dimensional_current)

        def current(t):
            return dimensional_current_eval.evaluate(t=t)

        function_list.append(current)

        standard_tests = StandardCurrentFunctionTests(function_list, always_array=True)
        standard_tests.test_all()

    def test_user_current(self):
        # create user-defined sin function
        def my_fun(t, A, omega):
            return A * np.sin(2 * np.pi * omega * t)

        # choose amplitude and frequency
        A = pybamm.electrical_parameters.I_typ
        omega = 3

        # pass my_fun to GetUserCurrent class, giving the additonal parameters as
        # keyword arguments
        current = pybamm.GetUserCurrent(my_fun, A=A, omega=omega)

        # set and process parameters
        parameter_values = pybamm.ParameterValues(
            {
                "Typical current [A]": 2,
                "Typical timescale [s]": 1,
                "Current function": current,
            }
        )
        dimensional_current = pybamm.electrical_parameters.dimensional_current_with_time
        dimensional_current_eval = parameter_values.process_symbol(dimensional_current)

        def user_current(t):
            return dimensional_current_eval.evaluate(t=t)

        # check output types
        standard_tests = StandardCurrentFunctionTests([user_current])
        standard_tests.test_all()

        # check output correct value
        time = np.linspace(0, 3600, 600)
        np.testing.assert_array_almost_equal(
            current(time), 2 * np.sin(2 * np.pi * 3 * time)
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
    unittest.main()
