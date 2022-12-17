#
# Tests for the Scalar class
#
import pybamm

import unittest
import numpy as np


class TestDDT(unittest.TestCase):
    def test_time_derivative(self):
        a = pybamm.Scalar(5).diff(pybamm.t)
        self.assertIsInstance(a, pybamm.Scalar)
        self.assertEqual(a.value, 0)

        a = pybamm.t.diff(pybamm.t)
        self.assertIsInstance(a, pybamm.Scalar)
        self.assertEqual(a.value, 1)

        a = (pybamm.t**2).diff(pybamm.t)
        self.assertEqual(a, (2 * pybamm.t**1 * 1))
        self.assertEqual(a.evaluate(t=1), 2)

        a = (2 + pybamm.t**2).diff(pybamm.t)
        self.assertEqual(a.evaluate(t=1), 2)

    def test_time_derivative_of_variable(self):

        a = (pybamm.Variable("a")).diff(pybamm.t)
        self.assertIsInstance(a, pybamm.VariableDot)
        self.assertEqual(a.name, "a'")

        p = pybamm.Parameter("p")
        a = 1 + p * pybamm.Variable("a")
        diff_a = a.diff(pybamm.t)
        self.assertIsInstance(diff_a, pybamm.Multiplication)
        self.assertEqual(diff_a.children[0].name, "p")
        self.assertEqual(diff_a.children[1].name, "a'")

        with self.assertRaises(pybamm.ModelError):
            a = (pybamm.Variable("a")).diff(pybamm.t).diff(pybamm.t)

    def test_time_derivative_of_state_vector(self):

        sv = pybamm.StateVector(slice(0, 10))
        y_dot = np.linspace(0, 2, 19)

        a = sv.diff(pybamm.t)
        self.assertIsInstance(a, pybamm.StateVectorDot)
        self.assertEqual(a.name[-1], "'")
        np.testing.assert_array_equal(
            a.evaluate(y_dot=y_dot), np.linspace(0, 1, 10)[:, np.newaxis]
        )

        with self.assertRaises(pybamm.ModelError):
            a = (sv).diff(pybamm.t).diff(pybamm.t)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
