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
    from julia.core import JuliaError

    # load julia libraries required for evaluating the strings
    Main.eval("using SparseArrays, LinearAlgebra")


@unittest.skipIf(not have_julia, "Julia not installed")
@unittest.skipIf(system() == "Windows", "Julia not supported on windows")
class TestEvaluate(unittest.TestCase):
    def evaluate_and_test_equal(
        self, expr, y_tests, t_tests=0.0, inputs=None, decimal=14, **kwargs
    ):
        if not isinstance(y_tests, list):
            y_tests = [y_tests]
        if not isinstance(t_tests, list):
            t_tests = [t_tests]
        if inputs is None:
            input_parameter_order = None
            p = 0.0
        else:
            input_parameter_order = list(inputs.keys())
            p = list(inputs.values())

        pybamm_eval = expr.evaluate(t=t_tests[0], y=y_tests[0], inputs=inputs).flatten()
        for preallocate in [True, False]:
            kwargs["funcname"] = (
                kwargs.get("funcname", "f") + "_" + str(int(preallocate))
            )
            evaluator_str = pybamm.get_julia_function(
                expr,
                input_parameter_order=input_parameter_order,
                preallocate=preallocate,
                **kwargs,
            )
            Main.eval(evaluator_str)
            funcname = kwargs.get("funcname", "f")
            Main.p = p
            for t_test, y_test in zip(t_tests, y_tests):
                Main.dy = np.zeros_like(pybamm_eval)
                Main.y = y_test
                Main.t = t_test
                try:
                    Main.eval(f"{funcname}!(dy,y,p,t)")
                except JuliaError as e:
                    # debugging
                    print(Main.dy, y_test, p, t_test)
                    print(evaluator_str)
                    raise e
                pybamm_eval = expr.evaluate(t=t_test, y=y_test, inputs=inputs).flatten()
                try:
                    np.testing.assert_array_almost_equal(
                        Main.dy.flatten(),
                        pybamm_eval,
                        decimal=decimal,
                    )
                except AssertionError as e:
                    # debugging
                    print(Main.dy, y_test, p, t_test)
                    print(evaluator_str)
                    raise e

    def test_exceptions(self):
        a = pybamm.Symbol("a")
        with self.assertRaisesRegex(NotImplementedError, "Conversion to Julia"):
            pybamm.get_julia_function(a)

    def test_evaluator_julia(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))

        y_tests = [np.array([[2], [3]]), np.array([[1], [3]])]
        t_tests = [1, 2]

        # test a * b
        expr = a * b
        self.evaluate_and_test_equal(expr, np.array([2.0, 3.0]))
        self.evaluate_and_test_equal(expr, np.array([1.0, 3.0]))

        # test function(a*b)
        expr = pybamm.cos(a * b)
        self.evaluate_and_test_equal(expr, np.array([1.0, 3.0]), funcname="g")

        # test a constant expression
        expr = pybamm.Multiplication(pybamm.Scalar(2), pybamm.Scalar(3))
        self.evaluate_and_test_equal(expr, 0.0)

        expr = pybamm.Multiplication(pybamm.Scalar(2), pybamm.Vector([1, 2, 3]))
        self.evaluate_and_test_equal(expr, None, funcname="g2")

        # test a larger expression
        expr = a * b + b + a ** 2 / b + 2 * a + b / 2 + 4
        self.evaluate_and_test_equal(expr, y_tests)

        # test something with time
        expr = a * pybamm.t
        self.evaluate_and_test_equal(expr, y_tests, t_tests=t_tests)

        # test something with a matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        expr = A @ pybamm.StateVector(slice(0, 2))
        self.evaluate_and_test_equal(expr, y_tests, funcname="g3")

        # test something with a heaviside
        a = pybamm.Vector([1, 2])
        expr = a <= pybamm.StateVector(slice(0, 2))
        self.evaluate_and_test_equal(expr, y_tests, funcname="g4")

        # test something with a minimum or maximum
        a = pybamm.Vector([1, 2])
        for expr in [
            pybamm.minimum(a, pybamm.StateVector(slice(0, 2))),
            pybamm.maximum(a, pybamm.StateVector(slice(0, 2))),
        ]:
            self.evaluate_and_test_equal(expr, y_tests, funcname="g5")

        # test something with an index
        expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
        self.evaluate_and_test_equal(expr, y_tests, funcname="g6")

        # test something with a sparse matrix multiplication
        A = pybamm.Matrix([[1, 2], [3, 4]])
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        expr = A @ B @ pybamm.StateVector(slice(0, 2))
        self.evaluate_and_test_equal(expr, y_tests, funcname="g7")

        expr = B @ pybamm.StateVector(slice(0, 2))
        self.evaluate_and_test_equal(expr, y_tests, funcname="g8")

        # test Inner
        expr = pybamm.Inner(pybamm.Vector([1, 2]), pybamm.StateVector(slice(0, 2)))
        self.evaluate_and_test_equal(expr, y_tests, funcname="g12")

        # test numpy concatenation
        a = pybamm.StateVector(slice(0, 3))
        b = pybamm.Vector([2, 3, 4])
        c = pybamm.Vector([5])

        y_tests = [np.array([[2], [3], [4]]), np.array([[1], [3], [2]])]

        expr = pybamm.NumpyConcatenation(a, b, c)
        self.evaluate_and_test_equal(expr, y_tests, funcname="g9")

        expr = pybamm.NumpyConcatenation(a, c)
        self.evaluate_and_test_equal(expr, y_tests, funcname="g10")

        # test sparse stack
        A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
        B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
        c = pybamm.StateVector(slice(0, 2))
        expr = pybamm.SparseStack(A, B) @ c
        self.evaluate_and_test_equal(expr, y_tests, funcname="g11")

    def test_evaluator_julia_input_parameters(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        c = pybamm.InputParameter("c")
        d = pybamm.InputParameter("d")

        # test one input parameter
        expr = a * c
        self.evaluate_and_test_equal(expr, np.array([2.0, 3.0]), inputs={"c": 5})

        # test several input parameters
        expr = a * c + b * d
        self.evaluate_and_test_equal(
            expr, np.array([2.0, 3.0]), inputs={"c": 5, "d": 6}
        )

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
            self.evaluate_and_test_equal(expr, y_test)

        for function in [
            pybamm.min,
            pybamm.max,
        ]:
            expr = function(a)
            self.evaluate_and_test_equal(expr, y_test)

        # More advanced tests for min
        b = pybamm.StateVector(slice(3, 6))
        concat = pybamm.NumpyConcatenation(2 * a, 3 * b)
        expr = pybamm.min(concat)
        self.evaluate_and_test_equal(expr, np.array([1, 2, 3, 4, 5, 6]), funcname="h1")

        v = pybamm.Vector([1, 2, 3])
        expr = pybamm.min(v * a)
        self.evaluate_and_test_equal(expr, y_test, funcname="h2")

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
        self.evaluate_and_test_equal(c_disc, y_tests)

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

        self.evaluate_and_test_equal(c_disc, y_tests)

    def test_evaluator_julia_discretised_operators(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        combined_submesh = mesh.combine_submeshes(*whole_cell)

        var = pybamm.Variable("var", domain=whole_cell)
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        disc.set_variable_slices([var])

        # grad
        grad_eqn = pybamm.grad(var)
        grad_eqn_disc = disc.process_symbol(grad_eqn)

        # div
        div_eqn = pybamm.div(var * grad_eqn)
        div_eqn_disc = disc.process_symbol(div_eqn)

        # test
        nodes = combined_submesh.nodes
        y_tests = [nodes ** 2 + 1, np.cos(nodes)]

        for i, expr in enumerate([grad_eqn_disc, div_eqn_disc]):
            self.evaluate_and_test_equal(expr, y_tests, funcname=f"f{i}", decimal=10)

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
            var: {
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
            self.evaluate_and_test_equal(expr, y_tests, funcname=f"f{i}", decimal=11)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
