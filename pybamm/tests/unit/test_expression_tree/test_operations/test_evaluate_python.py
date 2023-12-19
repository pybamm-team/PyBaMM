#
# Test for the evaluate-to-python functions
#
from tests import TestCase
import pybamm

from tests import get_discretisation_for_testing, get_1p1d_discretisation_for_testing
import unittest
import numpy as np
import scipy.sparse
from collections import OrderedDict

if pybamm.have_jax():
    import jax


def test_function(arg):
    return arg + arg


def test_function2(arg1, arg2):
    return arg1 + arg2


class TestEvaluate(TestCase):
    def test_find_symbols(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        # test a + b
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = a + b
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)

        # test keys of known_symbols
        self.assertEqual(next(iter(variable_symbols.keys())), a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)

        # test values of variable_symbols
        self.assertEqual(next(iter(variable_symbols.values())), "y[0:1]")
        self.assertEqual(list(variable_symbols.values())[1], "y[1:2]")

        var_a = pybamm.id_to_python_variable(a.id)
        var_b = pybamm.id_to_python_variable(b.id)
        self.assertEqual(
            list(variable_symbols.values())[2], f"{var_a} + {var_b}"
        )

        # test identical subtree
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = a + b + b
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)

        # test keys of variable_symbols
        self.assertEqual(next(iter(variable_symbols.keys())), a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.children[0].id)
        self.assertEqual(list(variable_symbols.keys())[3], expr.id)

        # test values of variable_symbols
        self.assertEqual(next(iter(variable_symbols.values())), "y[0:1]")
        self.assertEqual(list(variable_symbols.values())[1], "y[1:2]")
        self.assertEqual(
            list(variable_symbols.values())[2], f"{var_a} + {var_b}"
        )

        var_child = pybamm.id_to_python_variable(expr.children[0].id)
        self.assertEqual(
            list(variable_symbols.values())[3], f"{var_child} + {var_b}"
        )

        # test unary op
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = pybamm.maximum(a, -(b))
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)

        # test keys of variable_symbols
        self.assertEqual(next(iter(variable_symbols.keys())), a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.children[1].id)
        self.assertEqual(list(variable_symbols.keys())[3], expr.id)

        # test values of variable_symbols
        self.assertEqual(next(iter(variable_symbols.values())), "y[0:1]")
        self.assertEqual(list(variable_symbols.values())[1], "y[1:2]")
        self.assertEqual(list(variable_symbols.values())[2], f"-{var_b}")
        var_child = pybamm.id_to_python_variable(expr.children[1].id)
        self.assertEqual(
            list(variable_symbols.values())[3], f"np.maximum({var_a},{var_child})"
        )

        # test function
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = pybamm.Function(test_function, a)
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(next(iter(constant_symbols.keys())), expr.id)
        self.assertEqual(next(iter(constant_symbols.values())), test_function)
        self.assertEqual(next(iter(variable_symbols.keys())), a.id)
        self.assertEqual(list(variable_symbols.keys())[1], expr.id)
        self.assertEqual(next(iter(variable_symbols.values())), "y[0:1]")
        var_funct = pybamm.id_to_python_variable(expr.id, True)
        self.assertEqual(
            list(variable_symbols.values())[1], f"{var_funct}({var_a})"
        )

        # test matrix
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        A = pybamm.Matrix([[1, 2], [3, 4]])
        pybamm.find_symbols(A, constant_symbols, variable_symbols)
        self.assertEqual(len(variable_symbols), 0)
        self.assertEqual(next(iter(constant_symbols.keys())), A.id)
        np.testing.assert_allclose(
            next(iter(constant_symbols.values())), np.array([[1, 2], [3, 4]])
        )

        # test sparse matrix
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[0, 2], [0, 4]])))
        pybamm.find_symbols(A, constant_symbols, variable_symbols)
        self.assertEqual(len(variable_symbols), 0)
        self.assertEqual(next(iter(constant_symbols.keys())), A.id)
        np.testing.assert_allclose(
            next(iter(constant_symbols.values())).toarray(), A.entries.toarray()
        )

        # test numpy concatentate
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = pybamm.NumpyConcatenation(a, b)
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(next(iter(variable_symbols.keys())), a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)
        self.assertEqual(
            list(variable_symbols.values())[2],
            f"np.concatenate(({var_a},{var_b}))",
        )

        # test domain concatentate
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = pybamm.NumpyConcatenation(a, b)
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(next(iter(variable_symbols.keys())), a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)
        self.assertEqual(
            list(variable_symbols.values())[2],
            f"np.concatenate(({var_a},{var_b}))",
        )

        # test that Concatentation throws
        a = pybamm.StateVector(slice(0, 1), domain="test a")
        b = pybamm.StateVector(slice(1, 2), domain="test b")

        expr = pybamm.concatenation(a, b)
        with self.assertRaises(NotImplementedError):
            pybamm.find_symbols(expr, constant_symbols, variable_symbols)

        # test that these nodes throw
        for expr in (pybamm.Variable("a"), pybamm.Parameter("a")):
            with self.assertRaises(NotImplementedError):
                pybamm.find_symbols(expr, constant_symbols, variable_symbols)

    def test_domain_concatenation(self):
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        a_dom = ["negative electrode"]
        b_dom = ["positive electrode"]
        a_pts = mesh[a_dom[0]].npts
        b_pts = mesh[b_dom[0]].npts
        a = pybamm.StateVector(slice(0, a_pts), domain=a_dom)
        b = pybamm.StateVector(slice(a_pts, a_pts + b_pts), domain=b_dom)
        y = np.empty((a_pts + b_pts, 1))
        for i in range(len(y)):
            y[i] = i

        # concatenate
        expr = pybamm.DomainConcatenation([a, b], mesh)

        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(next(iter(variable_symbols.keys())), a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)

        var_a = pybamm.id_to_python_variable(a.id)
        var_b = pybamm.id_to_python_variable(b.id)
        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(
            list(variable_symbols.values())[2],
            f"np.concatenate(({var_a}[0:{a_pts}],{var_b}[0:{b_pts}]))",
        )

        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

        # check that concatenating a single domain is consistent
        expr = pybamm.DomainConcatenation([a], mesh)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

        # check the reordering in case a child vector has to be split up
        a_dom = ["separator"]
        b_dom = ["negative electrode", "positive electrode"]
        b0_pts = mesh[b_dom[0]].npts
        a0_pts = mesh[a_dom[0]].npts
        b1_pts = mesh[b_dom[1]].npts

        a = pybamm.StateVector(slice(0, a0_pts), domain=a_dom)
        b = pybamm.StateVector(slice(a0_pts, a0_pts + b0_pts + b1_pts), domain=b_dom)

        y = np.empty((a0_pts + b0_pts + b1_pts, 1))
        for i in range(len(y)):
            y[i] = i

        var_a = pybamm.id_to_python_variable(a.id)
        var_b = pybamm.id_to_python_variable(b.id)
        expr = pybamm.DomainConcatenation([a, b], mesh)
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)

        b0_str = f"{var_b}[0:{b0_pts}]"
        a0_str = f"{var_a}[0:{a0_pts}]"
        b1_str = f"{var_b}[{b0_pts}:{b0_pts + b1_pts}]"

        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(
            list(variable_symbols.values())[2],
            f"np.concatenate(({a0_str},{b0_str},{b1_str}))",
        )

        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

    def test_domain_concatenation_2D(self):
        disc = get_1p1d_discretisation_for_testing()

        a_dom = ["negative electrode"]
        b_dom = ["separator"]
        a = pybamm.Variable("a", domain=a_dom)
        b = pybamm.Variable("b", domain=b_dom)
        conc = pybamm.concatenation(2 * a, 3 * b)
        disc.set_variable_slices([a, b])
        expr = disc.process_symbol(conc)
        self.assertIsInstance(expr, pybamm.DomainConcatenation)

        y = np.empty((expr._size, 1))
        for i in range(len(y)):
            y[i] = i

        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)

        self.assertEqual(len(constant_symbols), 0)

        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

        # check that concatenating a single domain is consistent
        expr = disc.process_symbol(pybamm.concatenation(a))
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

    def test_to_python(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        # test a * b
        expr = a + b
        constant_str, variable_str = pybamm.to_python(expr)
        expected_str = (
            "var_[0-9m]+ = y\[0:1\].*\\n"
            "var_[0-9m]+ = y\[1:2\].*\\n"
            "var_[0-9m]+ = var_[0-9m]+ \+ var_[0-9m]+"
        )

        self.assertRegex(variable_str, expected_str)

    def test_evaluator_python(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        y_tests = [np.array([[2], [3]]), np.array([[1], [3]])]
        t_tests = [1, 2]

        # test a * b
        expr = a * b
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 6)
        result = evaluator(t=None, y=np.array([[1], [3]]))
        self.assertEqual(result, 3)

        # test function(a*b)
        expr = pybamm.Function(test_function, a * b)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 12)

        expr = pybamm.Function(test_function2, a, b)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 5)

        # test a constant expression
        expr = pybamm.Scalar(2) * pybamm.Scalar(3)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator()
        self.assertEqual(result, 6)

        # test a larger expression
        expr = a * b + b + a**2 / b + 2 * a + b / 2 + 4
        evaluator = pybamm.EvaluatorPython(expr)
        for y in y_tests:
            result = evaluator(t=None, y=y)
            self.assertEqual(result, expr.evaluate(t=None, y=y))

        # test something with time
        expr = a * pybamm.t
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        expr = A @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with a heaviside
        a = pybamm.Vector([1, 2])
        expr = a <= pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        expr = a > pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with a minimum or maximum
        a = pybamm.Vector([1, 2])
        expr = pybamm.minimum(a, pybamm.StateVector(slice(0, 2)))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        expr = pybamm.maximum(a, pybamm.StateVector(slice(0, 2)))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with an index
        expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a sparse matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        C = pybamm.Matrix(scipy.sparse.coo_matrix(np.array([[1, 0], [0, 4]])))
        expr = A @ B @ C @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        expr = B @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test numpy concatenation
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        c = pybamm.StateVector(slice(2, 3))

        y_tests = [np.array([[2], [3], [4]]), np.array([[1], [3], [2]])]
        t_tests = [1, 2]
        expr = pybamm.NumpyConcatenation(a, b)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))
        expr = pybamm.NumpyConcatenation(a, c)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test sparse stack
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        a = pybamm.StateVector(slice(0, 1))
        expr = pybamm.SparseStack(A, a * B)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y).toarray()
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())
        expr = pybamm.SparseStack(A)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator().toarray()
        np.testing.assert_allclose(result, expr.evaluate().toarray())

        # test Inner
        expr = pybamm.Inner(a, b)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        v = pybamm.StateVector(slice(0, 2))
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        for expr in [pybamm.Inner(A, v), pybamm.Inner(v, A)]:
            evaluator = pybamm.EvaluatorPython(expr)
            for t, y in zip(t_tests, y_tests):
                result = evaluator(t=t, y=y).toarray()
                np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

        y_tests = [np.array([[2], [3], [4], [5]]), np.array([[1], [3], [2], [1]])]
        t_tests = [1, 2]
        a = pybamm.StateVector(slice(0, 1), slice(3, 4))
        b = pybamm.StateVector(slice(1, 3))
        expr = a * b
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_find_symbols_jax(self):
        # test sparse conversion
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[0, 2], [0, 4]])))
        pybamm.find_symbols(A, constant_symbols, variable_symbols, output_jax=True)
        self.assertEqual(len(variable_symbols), 0)
        self.assertEqual(next(iter(constant_symbols.keys())), A.id)
        np.testing.assert_allclose(
            next(iter(constant_symbols.values())).toarray(), A.entries.toarray()
        )

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_evaluator_jax(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        y_tests = [
            np.array([[2.0], [3.0]]),
            np.array([[1.0], [3.0]]),
            np.array([1.0, 3.0]),
        ]
        t_tests = [1.0, 2.0]

        # test a * b
        expr = a * b
        evaluator = pybamm.EvaluatorJax(expr)
        result = evaluator(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 6)
        result = evaluator(t=None, y=np.array([[1], [3]]))
        self.assertEqual(result, 3)

        # test function(a*b)
        expr = pybamm.Function(test_function, a * b)
        evaluator = pybamm.EvaluatorJax(expr)
        result = evaluator(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 12)

        # test exp
        expr = pybamm.exp(a * b)
        evaluator = pybamm.EvaluatorJax(expr)
        result = evaluator(t=None, y=np.array([[2], [3]]))
        np.testing.assert_array_almost_equal(result, np.exp(6), decimal=15)

        # test a constant expression
        expr = pybamm.Scalar(2) * pybamm.Scalar(3)
        evaluator = pybamm.EvaluatorJax(expr)
        result = evaluator()
        self.assertEqual(result, 6)

        # test a larger expression
        expr = a * b + b + a**2 / b + 2 * a + b / 2 + 4
        evaluator = pybamm.EvaluatorJax(expr)
        for y in y_tests:
            result = evaluator(t=None, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=None, y=y))

        # test something with time
        expr = a * pybamm.t
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a matrix multiplication
        A = pybamm.Matrix(np.array([[1, 2], [3, 4]]))
        expr = A @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with a heaviside
        a = pybamm.Vector(np.array([1, 2]))
        expr = a <= pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        expr = a > pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with a minimum or maximum
        a = pybamm.Vector(np.array([1, 2]))
        expr = pybamm.minimum(a, pybamm.StateVector(slice(0, 2)))
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        expr = pybamm.maximum(a, pybamm.StateVector(slice(0, 2)))
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with an index
        expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a sparse matrix-vector multiplication
        A = pybamm.Matrix(np.array([[1, 2], [3, 4]]))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        C = pybamm.Matrix(scipy.sparse.coo_matrix(np.array([[1, 0], [0, 4]])))
        expr = A @ B @ C @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test the sparse-scalar multiplication
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        for expr in [
            A * pybamm.t @ pybamm.StateVector(slice(0, 2)),
            pybamm.t * A @ pybamm.StateVector(slice(0, 2)),
        ]:
            evaluator = pybamm.EvaluatorJax(expr)
            for t, y in zip(t_tests, y_tests):
                result = evaluator(t=t, y=y)
                np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test the sparse-scalar division
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        expr = A / (1.0 + pybamm.t) @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test sparse stack
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        a = pybamm.StateVector(slice(0, 1))
        expr = pybamm.SparseStack(A, a * B)
        with self.assertRaises(NotImplementedError):
            evaluator = pybamm.EvaluatorJax(expr)

        # test sparse mat-mat mult
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        a = pybamm.StateVector(slice(0, 1))
        expr = A @ (a * B)
        with self.assertRaises(NotImplementedError):
            evaluator = pybamm.EvaluatorJax(expr)

        # test numpy concatenation
        a = pybamm.Vector(np.array([[1], [2]]))
        b = pybamm.Vector(np.array([[3]]))
        expr = pybamm.NumpyConcatenation(a, b)
        evaluator = pybamm.EvaluatorJax(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test Inner
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1]])))
        v = pybamm.StateVector(slice(0, 1))
        for expr in [
            pybamm.Inner(A, v) @ v,
            pybamm.Inner(v, A) @ v,
            pybamm.Inner(v, v) @ v,
        ]:
            evaluator = pybamm.EvaluatorJax(expr)
            for t, y in zip(t_tests, y_tests):
                result = evaluator(t=t, y=y)
                np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_evaluator_jax_jacobian(self):
        a = pybamm.StateVector(slice(0, 1))
        y_tests = [np.array([[2.0]]), np.array([[1.0]]), np.array([1.0])]

        expr = a**2
        expr_jac = 2 * a
        evaluator = pybamm.EvaluatorJax(expr)
        evaluator_jac_test = evaluator.get_jacobian()
        evaluator_jac = pybamm.EvaluatorJax(expr_jac)
        for y in y_tests:
            result_test = evaluator_jac_test(t=None, y=y)
            result_true = evaluator_jac(t=None, y=y)
            np.testing.assert_allclose(result_test, result_true)

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_evaluator_jax_jvp(self):
        a = pybamm.StateVector(slice(0, 1))
        y_tests = [np.array([[2.0]]), np.array([[1.0]]), np.array([1.0])]
        v_tests = [np.array([[2.9]]), np.array([[0.9]]), np.array([1.3])]

        expr = a**2
        expr_jac = 2 * a
        evaluator = pybamm.EvaluatorJax(expr)
        evaluator_jac_test = evaluator.get_jacobian()
        evaluator_jac_action_test = evaluator.get_jacobian_action()
        evaluator_jac = pybamm.EvaluatorJax(expr_jac)
        for y, v in zip(y_tests, v_tests):
            result_test = evaluator_jac_test(t=None, y=y)
            result_test_times_v = evaluator_jac_action_test(t=None, y=y, v=v)
            result_true = evaluator_jac(t=None, y=y)
            result_true_times_v = evaluator_jac(t=None, y=y) @ v.reshape(-1, 1)
            np.testing.assert_allclose(result_test, result_true)
            np.testing.assert_allclose(result_test_times_v, result_true_times_v)

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_evaluator_jax_debug(self):
        a = pybamm.StateVector(slice(0, 1))
        expr = a**2
        y_test = np.array([2.0, 3.0])
        evaluator = pybamm.EvaluatorJax(expr)
        evaluator.debug(y=y_test)

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_evaluator_jax_inputs(self):
        a = pybamm.InputParameter("a")
        expr = a**2
        evaluator = pybamm.EvaluatorJax(expr)
        result = evaluator(inputs={"a": 2})
        self.assertEqual(result, 4)

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_jax_coo_matrix(self):
        A = pybamm.JaxCooMatrix([0, 1], [0, 1], [1.0, 2.0], (2, 2))
        Adense = jax.numpy.array([[1.0, 0], [0, 2.0]])
        v = jax.numpy.array([[2.0], [1.0]])

        np.testing.assert_allclose(A.toarray(), Adense)
        np.testing.assert_allclose(A @ v, Adense @ v)
        np.testing.assert_allclose(A.scalar_multiply(3.0).toarray(), Adense * 3.0)

        with self.assertRaises(NotImplementedError):
            A.multiply(v)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
