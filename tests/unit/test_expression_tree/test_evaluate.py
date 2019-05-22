#
# Test for the evaluate functions
#
import pybamm

import unittest
import numpy as np
import scipy.sparse
import os
from collections import OrderedDict


class TestEvaluate(unittest.TestCase):
    def test_find_symbols(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        # test a * b
        known_symbols = OrderedDict()
        expr = a * b
        pybamm.find_symbols(expr, known_symbols)

        # test keys of known_symbols
        self.assertEqual(list(known_symbols.keys())[0], a.id)
        self.assertEqual(list(known_symbols.keys())[1], b.id)
        self.assertEqual(list(known_symbols.keys())[2], expr.id)

        # test values of known_symbols
        self.assertEqual(list(known_symbols.values())[0], 'y[0:1]')
        self.assertEqual(list(known_symbols.values())[1], 'y[1:2]')

        var_a = pybamm.id_to_python_variable(a.id)
        var_b = pybamm.id_to_python_variable(b.id)
        self.assertEqual(list(known_symbols.values())[
                         2], '{} * {}'.format(var_a, var_b))

        # test identical subtree
        known_symbols = OrderedDict()
        expr = a * b * b
        pybamm.find_symbols(expr, known_symbols)

        # test keys of known_symbols
        self.assertEqual(list(known_symbols.keys())[0], a.id)
        self.assertEqual(list(known_symbols.keys())[1], b.id)
        self.assertEqual(list(known_symbols.keys())[2], expr.children[0].id)
        self.assertEqual(list(known_symbols.keys())[3], expr.id)

        # test values of known_symbols
        self.assertEqual(list(known_symbols.values())[0], 'y[0:1]')
        self.assertEqual(list(known_symbols.values())[1], 'y[1:2]')
        self.assertEqual(list(known_symbols.values())[
                         2], '{} * {}'.format(var_a, var_b))

        var_child = pybamm.id_to_python_variable(expr.children[0].id)
        self.assertEqual(list(known_symbols.values())[
                         3], '{} * {}'.format(var_child, var_b))

        # test unary op
        known_symbols = OrderedDict()
        expr = a * (-b)
        pybamm.find_symbols(expr, known_symbols)

        # test keys of known_symbols
        self.assertEqual(list(known_symbols.keys())[0], a.id)
        self.assertEqual(list(known_symbols.keys())[1], b.id)
        self.assertEqual(list(known_symbols.keys())[2], expr.children[1].id)
        self.assertEqual(list(known_symbols.keys())[3], expr.id)

        # test values of known_symbols
        self.assertEqual(list(known_symbols.values())[0], 'y[0:1]')
        self.assertEqual(list(known_symbols.values())[1], 'y[1:2]')
        self.assertEqual(list(known_symbols.values())[2], '-{}'.format(var_b))
        var_child = pybamm.id_to_python_variable(expr.children[1].id)
        self.assertEqual(list(known_symbols.values())[3],
                         '{} * {}'.format(var_a, var_child))

        # test matrix
        known_symbols = OrderedDict()
        A = pybamm.Matrix(np.array([[1, 2], [3, 4]]))
        pybamm.find_symbols(A, known_symbols)
        self.assertEqual(list(known_symbols.keys())[0], A.id)
        self.assertEqual(list(known_symbols.values())[0], 'np.array([[1,2],[3,4]])')

        # test sparse matrix
        known_symbols = OrderedDict()
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[0, 2], [0, 4]])))
        pybamm.find_symbols(A, known_symbols)
        self.assertEqual(list(known_symbols.keys())[0], A.id)
        self.assertEqual(list(known_symbols.values())[0],
                         'scipy.sparse.csr_matrix((np.array([2,4]), np.array([1,1]), np.array([0,1,2])))'
                         )

    def test_to_python(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        # test a * b
        expr = a * b
        funct_str = pybamm.to_python(expr)
        expected_str = \
            "var_[0-9m]+ = y\[0:1\]\\n" \
            "var_[0-9m]+ = y\[1:2\]\\n" \
            "var_[0-9m]+ = var_[0-9m]+ \* var_[0-9m]+"

        self.assertRegex(funct_str, expected_str)

    def test_evaluator_python(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        y_tests = [np.array([[2], [3]]), np.array([[1], [3]])]
        t_tests = [1, 2]

        # test a * b
        expr = a * b
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 6)
        result = evaluator.evaluate(t=None, y=np.array([[1], [3]]))
        self.assertEqual(result, 3)

        # test a larger expression
        expr = a * b + b + a**2 / b + 2*a + b/2 + 4
        evaluator = pybamm.EvaluatorPython(expr)
        for y in y_tests:
            result = evaluator.evaluate(t=None, y=y)
            self.assertEqual(result, expr.evaluate(t=None, y=y))

        # test something with time
        expr = a * pybamm.t
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a matrix multiplication
        A = pybamm.Matrix(np.array([[1, 2], [3, 4]]))
        expr = A @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with a sparse matrix multiplication
        A = pybamm.Matrix(np.array([[1, 2], [3, 4]]))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        expr = A @ B @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
