#
# Test for the evaluate-to-Julia functions
#
import pybamm

from tests import get_discretisation_for_testing, get_1p1d_discretisation_for_testing
import unittest
import numpy as np
import scipy.sparse
from collections import OrderedDict

from julia import Main


def test_function(arg):
    return arg + arg


def test_function2(arg1, arg2):
    return arg1 + arg2


class TestEvaluate(unittest.TestCase):
    def test_evaluator_julia(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        y_tests = [np.array([[2], [3]]), np.array([[1], [3]])]
        t_tests = [1, 2]

        # test a * b
        expr = a * b
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        result = evaluator(None, [2, 3], None)
        self.assertEqual(result, 6)
        result = evaluator(None, [1, 3], None)
        self.assertEqual(result, 3)

        # test function(a*b)
        expr = pybamm.cos(a * b)
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        result = evaluator(None, np.array([[2], [3]]), None)
        self.assertAlmostEqual(result, np.cos(6))

        # test a constant expression
        expr = pybamm.Scalar(2) * pybamm.Scalar(3)
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        result = evaluator(None, None, None)
        self.assertEqual(result, 6)

        # test a larger expression
        expr = a * b + b + a ** 2 / b + 2 * a + b / 2 + 4
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for y in y_tests:
            result = evaluator(None, y, None)
            self.assertEqual(result, expr.evaluate(t=None, y=y))

        # # test something with time
        # expr = a * pybamm.t
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     self.assertEqual(result, expr.evaluate(t=t, y=y))

        # # test something with a matrix multiplication
        # A = pybamm.Matrix([[1, 2], [3, 4]])
        # expr = A @ pybamm.StateVector(slice(0, 2))
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # # test something with a heaviside
        # a = pybamm.Vector([1, 2])
        # expr = a <= pybamm.StateVector(slice(0, 2))
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # expr = a > pybamm.StateVector(slice(0, 2))
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # # test something with a minimum or maximum
        # a = pybamm.Vector([1, 2])
        # expr = pybamm.minimum(a, pybamm.StateVector(slice(0, 2)))
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # expr = pybamm.maximum(a, pybamm.StateVector(slice(0, 2)))
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # # test something with an index
        # expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     self.assertEqual(result, expr.evaluate(t=t, y=y))

        # # test something with a sparse matrix multiplication
        # A = pybamm.Matrix([[1, 2], [3, 4]])
        # B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        # C = pybamm.Matrix(scipy.sparse.coo_matrix(np.array([[1, 0], [0, 4]])))
        # expr = A @ B @ C @ pybamm.StateVector(slice(0, 2))
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # expr = B @ pybamm.StateVector(slice(0, 2))
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # # test numpy concatenation
        # a = pybamm.StateVector(slice(0, 1))
        # b = pybamm.StateVector(slice(1, 2))
        # c = pybamm.StateVector(slice(2, 3))

        # y_tests = [np.array([[2], [3], [4]]), np.array([[1], [3], [2]])]
        # t_tests = [1, 2]
        # expr = pybamm.NumpyConcatenation(a, b)
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))
        # expr = pybamm.NumpyConcatenation(a, c)
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # # test sparse stack
        # A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        # B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        # a = pybamm.StateVector(slice(0, 1))
        # expr = pybamm.SparseStack(A, a * B)
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y).toarray()
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

        # # test Inner
        # expr = pybamm.Inner(a, b)
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # v = pybamm.StateVector(slice(0, 2))
        # A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        # expr = pybamm.Inner(A, v)
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y).toarray()
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

        # y_tests = [np.array([[2], [3], [4], [5]]), np.array([[1], [3], [2], [1]])]
        # t_tests = [1, 2]
        # a = pybamm.StateVector(slice(0, 1), slice(3, 4))
        # b = pybamm.StateVector(slice(1, 3))
        # expr = a * b
        # evaluator = pybamm.EvaluatorPython(expr)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator.evaluate(t=t, y=y)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

    def test_evaluator_julia_all_functions(self):
        a = pybamm.StateVector(slice(0, 1))
        y_test = np.array([1])

        for function in [
            pybamm.arcsinh,
            pybamm.cos,
            pybamm.cosh,
            pybamm.exp,
            pybamm.log,
            pybamm.log10,
            pybamm.sin,
            pybamm.sinh,
            pybamm.sqrt,
            pybamm.tanh,
            pybamm.arctan,
        ]:
            expr = function(a)
            evaluator_str = pybamm.get_julia_function(expr)
            evaluator = Main.eval(evaluator_str)
            result = evaluator(None, [1], None)
            self.assertAlmostEqual(result, expr.evaluate(y=y_test))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
