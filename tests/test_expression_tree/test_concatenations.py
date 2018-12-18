# Tests for the Concatenation class and subclasses
#
import pybamm

import numpy as np
import unittest


class TestConcatenations(unittest.TestCase):
    def test_base_concatenation(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        conc = pybamm.Concatenation(a, b, c, name="conc")
        self.assertEqual(conc.name, "conc")
        self.assertTrue(isinstance(conc.children[0], pybamm.Symbol))
        self.assertEqual(conc.children[0].name, "a")
        self.assertEqual(conc.children[1].name, "b")
        self.assertEqual(conc.children[2].name, "c")
        with self.assertRaises(NotImplementedError):
            conc.evaluate(None, 3)

    def test_numpy_concatenation_vectors(self):
        # with entries
        y = np.linspace(0, 1, 15)
        a = pybamm.Vector(y[:5])
        b = pybamm.Vector(y[5:9])
        c = pybamm.Vector(y[9:])
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(conc.evaluate(None, y), y)
        # with y_slice
        a = pybamm.StateVector(slice(0, 10))
        b = pybamm.StateVector(slice(10, 15))
        c = pybamm.StateVector(slice(15, 23))
        conc = pybamm.NumpyConcatenation(a, b, c)
        y = np.linspace(0, 1, 23)
        np.testing.assert_array_equal(conc.evaluate(None, y), y)

    def test_numpy_concatenation_vector_scalar(self):
        # with entries
        y = np.linspace(0, 1, 10)
        a = pybamm.Vector(y)
        b = pybamm.Scalar(16)
        c = pybamm.Scalar(3)
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(None, y), np.concatenate([y, np.array([16]), np.array([3])])
        )

        # with y_slice
        a = pybamm.StateVector(slice(0, 10))
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(None, y), np.concatenate([y, np.array([16]), np.array([3])])
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
