#
# Test for the evaluate-to-Julia functions
#
import pybamm

from tests import get_mesh_for_testing, get_1p1d_mesh_for_testing
import unittest
import numpy as np
import scipy.sparse
from platform import system

have_julia = pybamm.have_julia()
if have_julia and system() != "Windows":
    from julia.api import Julia

    Julia(compiled_modules=False)
    from julia import Main

    # load julia libraries required for evaluating the strings
    Main.eval("using SparseArrays, LinearAlgebra")


@unittest.skipIf(not have_julia, "Julia not installed")
@unittest.skipIf(system() == "Windows", "Julia not supported on windows")
class TestEvaluate(unittest.TestCase):
    def test_evaluator_julia(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        y_tests = [np.array([[2], [3]]), np.array([[1], [3]])]
        t_tests = [1, 2]

        # test a * b
        expr = a * b
        evaluator_str = pybamm.get_julia_function(expr)
        Main.eval(evaluator_str)
        Main.dy = [0.0]
        Main.y = np.array([2.0, 3.0])
        Main.eval("f!(dy,y,0,0)")
        self.assertEqual(Main.dy, 6)
        Main.dy = [0.0]
        Main.y = np.array([1.0, 3.0])
        Main.eval("f!(dy,y,0,0)")
        self.assertEqual(Main.dy, 3)

        # test function(a*b)
        expr = pybamm.cos(a * b)
        evaluator_str = pybamm.get_julia_function(expr, funcname="g")
        Main.eval(evaluator_str)
        Main.dy = [0.0]
        Main.y = np.array([2.0, 3.0])
        Main.eval("g!(dy,y,0,0)")
        self.assertAlmostEqual(Main.dy[0], np.cos(6), places=15)

        # test a constant expression
        expr = pybamm.Multiplication(pybamm.Scalar(2), pybamm.Scalar(3))
        evaluator_str = pybamm.get_julia_function(expr)
        Main.eval(evaluator_str)
        Main.dy = [0.0]
        Main.eval("f!(dy,y,0,0)")
        self.assertEqual(Main.dy, 6)

        expr = pybamm.Multiplication(pybamm.Scalar(2), pybamm.Vector([1, 2, 3]))
        evaluator_str = pybamm.get_julia_function(expr, funcname="g2")
        Main.eval(evaluator_str)
        Main.dy = [0.0] * 3
        Main.eval("g2!(dy,y,0,0)")
        np.testing.assert_array_equal(Main.dy, [2, 4, 6])

        # test a larger expression
        expr = a * b + b + a ** 2 / b + 2 * a + b / 2 + 4
        evaluator_str = pybamm.get_julia_function(expr)
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0]
            Main.y = y
            Main.eval("f!(dy,y,0,0)")
            self.assertEqual(Main.dy, expr.evaluate(t=None, y=y))

        # test something with time
        expr = a * pybamm.t
        evaluator_str = pybamm.get_julia_function(expr)
        Main.eval(evaluator_str)
        for t, y in zip(t_tests, y_tests):
            Main.dy = [0.0]
            Main.y = y
            Main.t = t
            Main.eval("f!(dy,y,0,t)")
            self.assertEqual(Main.dy, expr.evaluate(t=t, y=y))

        # test something with a matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        expr = A @ pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr, funcname="g3")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0]
            Main.y = y
            Main.eval("g3!(dy,y,0,0)")
            # note 1D arrays are flattened in Julia
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        # test something with a heaviside
        a = pybamm.Vector([1, 2])
        expr = a <= pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr, funcname="g4")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0]
            Main.y = y
            Main.eval("g4!(dy,y,0,0)")
            # note 1D arrays are flattened in Julia
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        # test something with a minimum or maximum
        a = pybamm.Vector([1, 2])
        for expr in [
            pybamm.minimum(a, pybamm.StateVector(slice(0, 2))),
            pybamm.maximum(a, pybamm.StateVector(slice(0, 2))),
        ]:
            evaluator_str = pybamm.get_julia_function(expr, funcname="g5")
            Main.eval(evaluator_str)
            for y in y_tests:
                Main.dy = [0.0, 0.0]
                Main.y = y
                Main.eval("g5!(dy,y,0,0)")
                np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        # test something with an index
        expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
        evaluator_str = pybamm.get_julia_function(expr, funcname="g6")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0]
            Main.y = y
            Main.eval("g6!(dy,y,0,0)")
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        # test something with a sparse matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        expr = A @ B @ pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr, funcname="g7")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0]
            Main.y = y
            Main.eval("g7!(dy,y,0,0)")
            # note 1D arrays are flattened in Julia
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        expr = B @ pybamm.StateVector(slice(0, 2))
        evaluator_str = pybamm.get_julia_function(expr, funcname="g8")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0]
            Main.y = y
            Main.eval("g8!(dy,y,0,0)")
            # note 1D arrays are flattened in Julia
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        # test numpy concatenation
        a = pybamm.StateVector(slice(0, 3))
        b = pybamm.Vector([2, 3, 4])
        c = pybamm.Vector([5])

        y_tests = [np.array([[2], [3], [4]]), np.array([[1], [3], [2]])]
        t_tests = [1, 2]

        expr = pybamm.NumpyConcatenation(a, b, c)
        evaluator_str = pybamm.get_julia_function(expr, funcname="g9")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            Main.y = y
            Main.eval("g9!(dy,y,0,0)")
            # note 1D arrays are flattened in Julia
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        expr = pybamm.NumpyConcatenation(a, c)
        evaluator_str = pybamm.get_julia_function(expr, funcname="g10")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0, 0.0, 0.0]
            Main.y = y
            Main.eval("g10!(dy,y,0,0)")
            # note 1D arrays are flattened in Julia
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        # test sparse stack
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        c = pybamm.StateVector(slice(0, 2))
        expr = pybamm.SparseStack(A, B) @ c
        evaluator_str = pybamm.get_julia_function(expr, funcname="g11")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0, 0.0, 0.0]
            Main.y = y
            Main.eval("g11!(dy,y,0,0)")
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

        # test Inner
        expr = pybamm.Inner(pybamm.Vector([1, 2]), pybamm.StateVector(slice(0, 2)))
        evaluator_str = pybamm.get_julia_function(expr, funcname="g12")
        Main.eval(evaluator_str)
        for y in y_tests:
            Main.dy = [0.0, 0.0]
            Main.y = y
            Main.eval("g12!(dy,y,0,0)")
            np.testing.assert_array_equal(Main.dy, expr.evaluate(y=y).flatten())

    def test_evaluator_julia_input_parameters(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        c = pybamm.InputParameter("c")
        d = pybamm.InputParameter("d")

        # test one input parameter: a * c
        expr = a * c
        evaluator_str = pybamm.get_julia_function(expr, input_parameter_order=["c"])
        Main.eval(evaluator_str)
        Main.dy = [0.0]
        Main.y = np.array([2.0, 3.0])
        Main.p = [5]
        Main.eval("f!(dy,y,p,0)")
        self.assertEqual(Main.dy, 10)

        # test several input parameters: a * c + b * d
        expr = a * c + b * d
        evaluator_str = pybamm.get_julia_function(
            expr, input_parameter_order=["c", "d"]
        )
        Main.eval(evaluator_str)
        Main.dy = [0.0]
        Main.y = np.array([2.0, 3.0])
        Main.p = [5, 6]
        Main.eval("f!(dy,y,p,0)")
        self.assertEqual(Main.dy, 28)

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
            Main.eval(evaluator_str)
            Main.dy = 0.0 * y_test
            Main.y = y_test
            Main.eval("f!(dy,y,0,0)")
            np.testing.assert_almost_equal(
                Main.dy, expr.evaluate(y=y_test).flatten(), decimal=14
            )

        for function in [
            pybamm.min,
            pybamm.max,
        ]:
            expr = function(a)
            evaluator_str = pybamm.get_julia_function(expr)
            Main.eval(evaluator_str)
            Main.dy = [0.0]
            Main.y = y_test
            Main.eval("f!(dy,y,0,0)")
            np.testing.assert_equal(Main.dy, expr.evaluate(y=y_test).flatten())

    def test_evaluator_julia_domain_concatenation(self):
        c_n = pybamm.Variable("c_n", domain="negative electrode")
        c_s = pybamm.Variable("c_s", domain="separator")
        c_p = pybamm.Variable("c_p", domain="positive electrode")
        c = pybamm.concatenation(c_n / 2, c_s / 3, c_p / 4)
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*c.domain)
        nodes = combined_submesh.nodes
        y_tests = [nodes ** 2 + 1, np.cos(nodes)]

        # discretise and evaluate the variable
        disc.set_variable_slices([c_n, c_s, c_p])
        c_disc = disc.process_symbol(c)

        evaluator_str = pybamm.get_julia_function(c_disc)
        Main.eval(evaluator_str)
        for y_test in y_tests:
            pybamm_eval = c_disc.evaluate(y=y_test).flatten()
            Main.dy = np.zeros_like(pybamm_eval)
            Main.y = y_test
            Main.eval("f!(dy,y,0,0)")
            np.testing.assert_equal(Main.dy, pybamm_eval)

    def test_evaluator_julia_domain_concatenation_2D(self):
        c_n = pybamm.Variable(
            "c_n",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        c_s = pybamm.Variable(
            "c_s",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        c_p = pybamm.Variable(
            "c_p",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        c = pybamm.concatenation(c_n / 2, c_s / 3, c_p / 4)
        # create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*c.domain)
        nodes = np.linspace(
            0, 1, combined_submesh.npts * mesh["current collector"].npts
        )
        y_tests = [nodes ** 2 + 1, np.cos(nodes)]

        # discretise and evaluate the variable
        disc.set_variable_slices([c_n, c_s, c_p])
        c_disc = disc.process_symbol(c)

        evaluator_str = pybamm.get_julia_function(c_disc)
        Main.eval(evaluator_str)
        for y_test in y_tests:
            pybamm_eval = c_disc.evaluate(y=y_test).flatten()
            Main.dy = np.zeros_like(pybamm_eval)
            Main.y = y_test
            Main.eval("f!(dy,y,0,0)")
            np.testing.assert_equal(Main.dy, pybamm_eval)

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

        for i, expr in enumerate([grad_eqn_disc, div_eqn_disc]):
            evaluator_str = pybamm.get_julia_function(expr, funcname=f"f{i}")
            Main.eval(evaluator_str)
            for y_test in y_tests:
                pybamm_eval = expr.evaluate(y=y_test).flatten()
                Main.dy = np.zeros_like(pybamm_eval)
                Main.y = y_test
                Main.eval(f"f{i}!(dy,y,0,0)")
                np.testing.assert_almost_equal(Main.dy, pybamm_eval, decimal=7)

        # Test without preallocation
        for i, expr in enumerate([grad_eqn_disc, div_eqn_disc]):
            evaluator_str = pybamm.get_julia_function(
                expr, funcname=f"f{i+10}", preallocate=False
            )
            Main.eval(evaluator_str)
            for y_test in y_tests:
                pybamm_eval = expr.evaluate(y=y_test).flatten()
                Main.dy = np.zeros_like(pybamm_eval)
                Main.y = y_test
                Main.eval(f"f{i}!(dy,y,0,0)")
                np.testing.assert_almost_equal(Main.dy, pybamm_eval, decimal=7)

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

        for i, expr in enumerate([grad_eqn_disc, div_eqn_disc]):
            evaluator_str = pybamm.get_julia_function(expr, funcname=f"f{i}")
            Main.eval(evaluator_str)
            for y_test in y_tests:
                pybamm_eval = expr.evaluate(y=y_test).flatten()
                Main.dy = np.zeros_like(pybamm_eval)
                Main.y = y_test
                Main.eval(f"f{i}!(dy,y,0,0)")
                np.testing.assert_almost_equal(Main.dy, pybamm_eval, decimal=7)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
