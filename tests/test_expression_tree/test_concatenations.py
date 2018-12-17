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
        self.assertEqual(conc.children, (a, b, c))
        self.assertEqual(a.parent, conc)
        with self.assertRaises(NotImplementedError):
            conc.evaluate(3)

    def test_numpy_concatenation_vectors(self):
        # with entries
        y = np.linspace(0, 1, 15)
        a = pybamm.Vector(y[:5])
        b = pybamm.Vector(y[5:9])
        c = pybamm.Vector(y[9:])
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(conc.evaluate(y), y)
        # with y_slice
        a = pybamm.Vector(slice(0, 10))
        b = pybamm.Vector(slice(10, 15))
        c = pybamm.Vector(slice(15, 23))
        conc = pybamm.NumpyConcatenation(a, b, c)
        y = np.linspace(0, 1, 23)
        np.testing.assert_array_equal(conc.evaluate(y), y)

    def test_numpy_concatenation_vector_scalar(self):
        # with entries
        y = np.linspace(0, 1, 10)
        a = pybamm.Vector(y)
        b = pybamm.Scalar(16)
        c = pybamm.Scalar(3)
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(y), np.concatenate([y, np.array([16]), np.array([3])])
        )

        # with y_slice
        a = pybamm.Vector(slice(0, 10))
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(y), np.concatenate([y, np.array([16]), np.array([3])])
        )
