#
# Test for the evaluate functions
#
import pybamm

from tests import get_discretisation_for_testing, get_1p1d_discretisation_for_testing
import unittest
import numpy as np
import scipy.sparse
from collections import OrderedDict


def test_function(arg):
    return arg + arg


class TestEvaluate(unittest.TestCase):
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
        self.assertEqual(list(variable_symbols.keys())[0], a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)

        # test values of variable_symbols
        self.assertEqual(list(variable_symbols.values())[0], "y[:1][[True]]")
        self.assertEqual(list(variable_symbols.values())[1], "y[:2][[False, True]]")

        var_a = pybamm.id_to_python_variable(a.id)
        var_b = pybamm.id_to_python_variable(b.id)
        self.assertEqual(
            list(variable_symbols.values())[2], "{} + {}".format(var_a, var_b)
        )

        # test identical subtree
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = a + b + b
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)

        # test keys of variable_symbols
        self.assertEqual(list(variable_symbols.keys())[0], a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.children[0].id)
        self.assertEqual(list(variable_symbols.keys())[3], expr.id)

        # test values of variable_symbols
        self.assertEqual(list(variable_symbols.values())[0], "y[:1][[True]]")
        self.assertEqual(list(variable_symbols.values())[1], "y[:2][[False, True]]")
        self.assertEqual(
            list(variable_symbols.values())[2], "{} + {}".format(var_a, var_b)
        )

        var_child = pybamm.id_to_python_variable(expr.children[0].id)
        self.assertEqual(
            list(variable_symbols.values())[3], "{} + {}".format(var_child, var_b)
        )

        # test unary op
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = a + (-b)
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)

        # test keys of variable_symbols
        self.assertEqual(list(variable_symbols.keys())[0], a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.children[1].id)
        self.assertEqual(list(variable_symbols.keys())[3], expr.id)

        # test values of variable_symbols
        self.assertEqual(list(variable_symbols.values())[0], "y[:1][[True]]")
        self.assertEqual(list(variable_symbols.values())[1], "y[:2][[False, True]]")
        self.assertEqual(list(variable_symbols.values())[2], "-{}".format(var_b))
        var_child = pybamm.id_to_python_variable(expr.children[1].id)
        self.assertEqual(
            list(variable_symbols.values())[3], "{} + {}".format(var_a, var_child)
        )

        # test function
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = pybamm.Function(test_function, a)
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(list(constant_symbols.keys())[0], expr.id)
        self.assertEqual(list(constant_symbols.values())[0], test_function)
        self.assertEqual(list(variable_symbols.keys())[0], a.id)
        self.assertEqual(list(variable_symbols.keys())[1], expr.id)
        self.assertEqual(list(variable_symbols.values())[0], "y[:1][[True]]")
        var_funct = pybamm.id_to_python_variable(expr.id, True)
        self.assertEqual(
            list(variable_symbols.values())[1], "{}({})".format(var_funct, var_a)
        )

        # test matrix
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        A = pybamm.Matrix(np.array([[1, 2], [3, 4]]))
        pybamm.find_symbols(A, constant_symbols, variable_symbols)
        self.assertEqual(len(variable_symbols), 0)
        self.assertEqual(list(constant_symbols.keys())[0], A.id)
        np.testing.assert_allclose(
            list(constant_symbols.values())[0], np.array([[1, 2], [3, 4]])
        )

        # test sparse matrix
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[0, 2], [0, 4]])))
        pybamm.find_symbols(A, constant_symbols, variable_symbols)
        self.assertEqual(len(variable_symbols), 0)
        self.assertEqual(list(constant_symbols.keys())[0], A.id)
        np.testing.assert_allclose(
            list(constant_symbols.values())[0].toarray(), A.entries.toarray()
        )

        # test numpy concatentate
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = pybamm.NumpyConcatenation(a, b)
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(list(variable_symbols.keys())[0], a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)
        self.assertEqual(
            list(variable_symbols.values())[2],
            "np.concatenate(({},{}))".format(var_a, var_b),
        )

        # test domain concatentate
        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        expr = pybamm.NumpyConcatenation(a, b)
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(list(variable_symbols.keys())[0], a.id)
        self.assertEqual(list(variable_symbols.keys())[1], b.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)
        self.assertEqual(
            list(variable_symbols.values())[2],
            "np.concatenate(({},{}))".format(var_a, var_b),
        )

        # test that Concatentation throws
        expr = pybamm.Concatenation(a, b)
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

        # concatenate them the "wrong" way round to check they get reordered correctly
        expr = pybamm.DomainConcatenation([b, a], mesh)

        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)
        self.assertEqual(list(variable_symbols.keys())[0], b.id)
        self.assertEqual(list(variable_symbols.keys())[1], a.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)

        var_a = pybamm.id_to_python_variable(a.id)
        var_b = pybamm.id_to_python_variable(b.id)
        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(
            list(variable_symbols.values())[2],
            "np.concatenate(({}[0:{}],{}[0:{}]))".format(var_a, a_pts, var_b, b_pts),
        )

        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

        # check that concatenating a single domain is consistent
        expr = pybamm.DomainConcatenation([a], mesh)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate(y=y)
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

        b0_str = "{}[0:{}]".format(var_b, b0_pts)
        a0_str = "{}[0:{}]".format(var_a, a0_pts)
        b1_str = "{}[{}:{}]".format(var_b, b0_pts, b0_pts + b1_pts)

        self.assertEqual(len(constant_symbols), 0)
        self.assertEqual(
            list(variable_symbols.values())[2],
            "np.concatenate(({},{},{}))".format(b0_str, a0_str, b1_str),
        )

        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

    def test_domain_concatenation_2D(self):
        disc = get_1p1d_discretisation_for_testing()

        a_dom = ["negative electrode"]
        b_dom = ["separator"]
        a = pybamm.Variable("a", domain=a_dom)
        b = pybamm.Variable("b", domain=b_dom)
        conc = pybamm.Concatenation(a, b)
        disc.set_variable_slices([conc])
        expr = disc.process_symbol(conc)
        self.assertIsInstance(expr, pybamm.DomainConcatenation)
        a_disc = expr.children[0]
        b_disc = expr.children[1]

        y = np.empty((expr._size, 1))
        for i in range(len(y)):
            y[i] = i

        constant_symbols = OrderedDict()
        variable_symbols = OrderedDict()
        pybamm.find_symbols(expr, constant_symbols, variable_symbols)

        self.assertEqual(list(variable_symbols.keys())[0], a_disc.id)
        self.assertEqual(list(variable_symbols.keys())[1], b_disc.id)
        self.assertEqual(list(variable_symbols.keys())[2], expr.id)

        self.assertEqual(len(constant_symbols), 0)

        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

        # check that concatenating a single domain is consistent
        expr = disc.process_symbol(pybamm.Concatenation(a))
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate(y=y)
        np.testing.assert_allclose(result, expr.evaluate(y=y))

    def test_to_python(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        # test a * b
        expr = a + b
        constant_str, variable_str = pybamm.to_python(expr)
        expected_str = (
            "self\.var_[0-9m]+ = y\[:1\]\[\[True\]\].*\\n"
            "self\.var_[0-9m]+ = y\[:2\]\[\[False, True\]\].*\\n"
            "self\.var_[0-9m]+ = self\.var_[0-9m]+ \+ self\.var_[0-9m]+"
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
        result = evaluator.evaluate(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 6)
        result = evaluator.evaluate(t=None, y=np.array([[1], [3]]))
        self.assertEqual(result, 3)

        # test function(a*b)
        expr = pybamm.Function(test_function, a * b)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate(t=None, y=np.array([[2], [3]]))
        self.assertEqual(result, 12)

        # test a constant expression
        expr = pybamm.Scalar(2) * pybamm.Scalar(3)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate()
        self.assertEqual(result, 6)

        # test a larger expression
        expr = a * b + b + a ** 2 / b + 2 * a + b / 2 + 4
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

        # test something with a heaviside
        a = pybamm.Vector(np.array([1, 2]))
        expr = a <= pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        expr = a > pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with a minimum or maximum
        a = pybamm.Vector(np.array([1, 2]))
        expr = pybamm.minimum(a, pybamm.StateVector(slice(0, 2)))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        expr = pybamm.maximum(a, pybamm.StateVector(slice(0, 2)))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test something with an index
        expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a sparse matrix multiplication
        A = pybamm.Matrix(np.array([[1, 2], [3, 4]]))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        C = pybamm.Matrix(scipy.sparse.coo_matrix(np.array([[1, 0], [0, 4]])))
        expr = A @ B @ C @ pybamm.StateVector(slice(0, 2))
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test numpy concatenation
        a = pybamm.Vector(np.array([[1], [2]]))
        b = pybamm.Vector(np.array([[3]]))
        expr = pybamm.NumpyConcatenation(a, b)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y)
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # test sparse stack
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        expr = pybamm.SparseStack(A, B)
        evaluator = pybamm.EvaluatorPython(expr)
        for t, y in zip(t_tests, y_tests):
            result = evaluator.evaluate(t=t, y=y).toarray()
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

        # test Inner
        v = pybamm.Vector(np.ones(5), domain="test")
        w = pybamm.Vector(2 * np.ones(5), domain="test")
        expr = pybamm.Inner(v, w)
        evaluator = pybamm.EvaluatorPython(expr)
        result = evaluator.evaluate()
        np.testing.assert_allclose(result, expr.evaluate())


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
