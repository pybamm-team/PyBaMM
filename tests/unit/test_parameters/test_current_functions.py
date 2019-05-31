#
# Tests for current input functions
#
import pybamm.parameters.standard_current_functions as cf
import numbers

import unittest
import numpy as np


class TestCurrentFunctions(unittest.TestCase):
    def test_all_functions(self):
        function_list = [cf.sin_current]
        standard_tests = StandardCurrentFunctionTests(function_list)
        standard_tests.test_all()


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
