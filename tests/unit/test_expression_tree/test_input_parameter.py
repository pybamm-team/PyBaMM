#
# Tests for the InputParameter class
#
import numpy as np
import pybamm
import unittest


class TestInputParameter(unittest.TestCase):
    def test_input_parameter_init(self):
        a = pybamm.InputParameter("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.evaluate(inputs={"a": 1}), 1)
        self.assertEqual(a.evaluate(inputs={"a": 5}), 5)

    def test_set_expected_size(self):
        a = pybamm.InputParameter("a")
        a.set_expected_size(10)
        self.assertEqual(a._expected_size, 10)
        y = np.linspace(0, 1, 10)
        np.testing.assert_array_equal(a.evaluate(inputs={"a": y}), y)
        with self.assertRaisesRegex(
            ValueError,
            "Input parameter 'a' was given an object of size '1' but was expecting an "
            "object of size '10'",
        ):
            a.evaluate(inputs={"a": 5})

    def test_evaluate_for_shape(self):
        a = pybamm.InputParameter("a")
        self.assertTrue(np.isnan(a.evaluate_for_shape()))

    def test_errors(self):
        a = pybamm.InputParameter("a")
        with self.assertRaises(TypeError):
            a.evaluate(inputs="not a dictionary")
        with self.assertRaises(KeyError):
            a.evaluate(inputs={"bad param": 5})
        # if u is not provided it gets turned into a dictionary and then raises KeyError
        with self.assertRaises(KeyError):
            a.evaluate()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
