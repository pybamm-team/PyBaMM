#
# Tests for the InputParameter class
#
import numbers
import pybamm
import unittest


class TestInputParameter(unittest.TestCase):
    def test_input_parameter_init(self):
        a = pybamm.InputParameter("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.evaluate(u={"a": 1}), 1)
        self.assertEqual(a.evaluate(u={"a": 5}), 5)

    def test_evaluate_for_shape(self):
        a = pybamm.InputParameter("a")
        self.assertIsInstance(a.evaluate_for_shape(), numbers.Number)

    def test_errors(self):
        a = pybamm.InputParameter("a")
        with self.assertRaises(TypeError):
            a.evaluate(u="not a dictionary")
        with self.assertRaises(KeyError):
            a.evaluate(u={"bad param": 5})
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
