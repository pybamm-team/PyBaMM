#
# Test for the evaluate-to-Julia functions
#
import pybamm

from tests import (
    get_mesh_for_testing,
    get_1p1d_mesh_for_testing,
    get_discretisation_for_testing,
    get_1p1d_discretisation_for_testing,
)
import unittest
import numpy as np
import scipy.sparse
from collections import OrderedDict

from julia import Main


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

        # test something with time
        expr = a * pybamm.t
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        expr = A @ pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            # note 1D arrays are flattened in Julia
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

        # test something with a heaviside
        a = pybamm.Vector([1, 2])
        expr = a <= pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            # note 1D arrays are flattened in Julia
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

        expr = a > pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            # note 1D arrays are flattened in Julia
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

        # # test something with a minimum or maximum
        # a = pybamm.Vector([1, 2])
        # expr = pybamm.minimum(a, pybamm.StateVector(slice(0, 2)))
        # evaluator_str = pybamm.get_julia_function(expr)
        # evaluator = Main.eval(evaluator_str)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator(t,y,None)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # expr = pybamm.maximum(a, pybamm.StateVector(slice(0, 2)))
        # evaluator_str = pybamm.get_julia_function(expr)
        # evaluator = Main.eval(evaluator_str)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator(t,y,None)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # # test something with an index
        # expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
        # evaluator_str = pybamm.get_julia_function(expr)
        # evaluator = Main.eval(evaluator_str)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator(t,y,None)
        #     self.assertEqual(result, expr.evaluate(t=t, y=y))

        # test something with a sparse matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        C = pybamm.Matrix(scipy.sparse.coo_matrix(np.array([[1, 0], [0, 4]])))
        expr = A @ B @ C @ pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            # note 1D arrays are flattened in Julia
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

        expr = B @ pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            # note 1D arrays are flattened in Julia
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

        # test numpy concatenation
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        c = pybamm.StateVector(slice(2, 3))

        y_tests = [np.array([[2], [3], [4]]), np.array([[1], [3], [2]])]
        t_tests = [1, 2]

        expr = pybamm.NumpyConcatenation(a, b)
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            # note 1D arrays are flattened in Julia
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

        expr = pybamm.NumpyConcatenation(a, c)
        evaluator_str = pybamm.get_julia_function(expr)
        evaluator = Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            result = evaluator(t, y, None)
            # note 1D arrays are flattened in Julia
            np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

        # # test sparse stack
        # A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        # B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        # a = pybamm.StateVector(slice(0, 1))
        # expr = pybamm.SparseStack(A, a * B)
        # evaluator_str = pybamm.get_julia_function(expr)
        # evaluator = Main.eval(evaluator_str)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator(t,y,None).toarray()
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

        # # test Inner
        # expr = pybamm.Inner(a, b)
        # evaluator_str = pybamm.get_julia_function(expr)
        # evaluator = Main.eval(evaluator_str)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator(t,y,None)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

        # v = pybamm.StateVector(slice(0, 2))
        # A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        # expr = pybamm.Inner(A, v)
        # evaluator_str = pybamm.get_julia_function(expr)
        # evaluator = Main.eval(evaluator_str)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator(t,y,None).toarray()
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

        # y_tests = [np.array([[2], [3], [4], [5]]), np.array([[1], [3], [2], [1]])]
        # t_tests = [1, 2]
        # a = pybamm.StateVector(slice(0, 1), slice(3, 4))
        # b = pybamm.StateVector(slice(1, 3))
        # expr = a * b
        # evaluator_str = pybamm.get_julia_function(expr)
        # evaluator = Main.eval(evaluator_str)
        # for t, y in zip(t_tests, y_tests):
        #     result = evaluator(t,y,None)
        #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

    def test_evaluator_julia_all_functions(self):
        a = pybamm.StateVector(slice(0, 3))
        y_test = np.array([1, 2, 3])

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
            result = evaluator(None, y_test, None)
            np.testing.assert_almost_equal(result, expr.evaluate(y=y_test).flatten())

    def test_evaluator_julia_discretised_operators(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # grad
        var = pybamm.Variable("var", domain=whole_cell)
        grad_eqn = pybamm.grad(var)
        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        # div: test on linear y (should have laplacian zero) so change bcs
        div_eqn = pybamm.div(var * grad_eqn)

        div_eqn_disc = disc.process_symbol(div_eqn)

        # test
        nodes = combined_submesh.nodes
        y_tests = [nodes ** 2 + 1, np.cos(nodes)]

        for expr in [grad_eqn_disc, div_eqn_disc]:
            for y_test in y_tests:
                evaluator_str = pybamm.get_julia_function(expr)
                evaluator = Main.eval(evaluator_str)
                result = evaluator(None, y_test, None)
                np.testing.assert_almost_equal(
                    result, expr.evaluate(y=y_test).flatten()
                )

    def test_evaluator_julia_discretised_microscale(self):
        # create discretisation
        mesh = get_1p1d_mesh_for_testing(xpts=5, rpts=5, zpts=2)
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        submesh = mesh["negative particle"]

        # grad
        # grad(r) == 1
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(var * grad_eqn)

        boundary_conditions = {
            var.id: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }

        disc.bcs = boundary_conditions

        disc.set_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # test
        total_npts = (
            submesh.npts
            * mesh["negative electrode"].npts
            * mesh["current collector"].npts
        )
        y_tests = [np.linspace(0, 1, total_npts) ** 2]

        for expr in [div_eqn_disc]:
            for y_test in y_tests:
                evaluator_str = pybamm.get_julia_function(expr)
                evaluator = Main.eval(evaluator_str)
                result = evaluator(None, y_test, None)
                np.testing.assert_almost_equal(
                    result, expr.evaluate(y=y_test).flatten()
                )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
