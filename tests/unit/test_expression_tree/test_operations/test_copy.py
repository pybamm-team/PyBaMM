#
# Test for making copies
#
from tests import TestCase
import numpy as np
import pybamm
import unittest
from tests import get_mesh_for_testing


class TestCopy(TestCase):
    def test_symbol_new_copy(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.IndependentVariable("Variable_c")
        v_n = pybamm.Variable("v", "negative electrode")
        v_n_2D = pybamm.Variable(
            "v",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        x_n = pybamm.standard_spatial_vars.x_n
        v_s = pybamm.Variable("v", "separator")
        vec = pybamm.Vector([1, 2, 3, 4, 5])
        mat = pybamm.Matrix([[1, 2], [3, 4]])
        mesh = get_mesh_for_testing()

        for symbol in [
            a + b,
            a - b,
            a * b,
            a / b,
            a**b,
            -a,
            abs(a),
            c,
            pybamm.Function(np.sin, a),
            pybamm.FunctionParameter("function", {"a": a}),
            pybamm.grad(v_n),
            pybamm.div(pybamm.grad(v_n)),
            pybamm.upwind(v_n),
            pybamm.IndefiniteIntegral(v_n, x_n),
            pybamm.BackwardIndefiniteIntegral(v_n, x_n),
            pybamm.BoundaryValue(v_n, "right"),
            pybamm.BoundaryGradient(v_n, "right"),
            pybamm.PrimaryBroadcast(a, "domain"),
            pybamm.SecondaryBroadcast(v_n, "current collector"),
            pybamm.TertiaryBroadcast(v_n_2D, "current collector"),
            pybamm.FullBroadcast(a, "domain", {"secondary": "other domain"}),
            pybamm.concatenation(v_n, v_s),
            pybamm.NumpyConcatenation(a, b, v_s),
            pybamm.DomainConcatenation([v_n, v_s], mesh),
            pybamm.Parameter("param"),
            pybamm.InputParameter("param"),
            pybamm.StateVector(slice(0, 56)),
            pybamm.Matrix(np.ones((50, 40))),
            pybamm.SpatialVariable("x", ["negative electrode"]),
            pybamm.t,
            pybamm.Index(vec, 1),
            pybamm.NotConstant(a),
            pybamm.minimum(a, b),
            pybamm.maximum(a, b),
            pybamm.SparseStack(mat, mat),
            pybamm.Equality(a, b),
            pybamm.EvaluateAt(a, 0),
        ]:
            self.assertEqual(symbol, symbol.new_copy())

    def test_symbol_new_copy_new_children(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")

        # binary operations
        for symbol_ab, symbol_ba in zip(
            [
                a + b,
                a - b,
                a * b,
                a / b,
                a**b,
                pybamm.minimum(a, b),
                pybamm.maximum(a, b),
                pybamm.Equality(a, b),
            ],
            [
                b + a,
                b - a,
                b * a,
                b / a,
                b**a,
                pybamm.minimum(b, a),
                pybamm.maximum(b, a),
                pybamm.Equality(b, a),
            ],
        ):
            self.assertEqual(symbol_ab.new_copy(new_children=[b, a]), symbol_ba)

        # unary operations
        for symbol_a, symbol_b in zip(
            [
                -a,
                abs(a),
                pybamm.Function(np.sin, a),
                pybamm.PrimaryBroadcast(a, "domain"),
                pybamm.FullBroadcast(a, "domain", {"secondary": "other domain"}),
                pybamm.NotConstant(a),
                pybamm.EvaluateAt(a, 0),
            ],
            [
                -b,
                abs(b),
                pybamm.Function(np.sin, b),
                pybamm.PrimaryBroadcast(b, "domain"),
                pybamm.FullBroadcast(b, "domain", {"secondary": "other domain"}),
                pybamm.NotConstant(b),
                pybamm.EvaluateAt(b, 0),
            ],
        ):
            self.assertEqual(symbol_a.new_copy(new_children=[b]), symbol_b)

        v_n = pybamm.Variable("v", "negative electrode")
        w_n = pybamm.Variable("w", "negative electrode")
        x_n = pybamm.standard_spatial_vars.x_n

        for symbol_v, symbol_w in zip(
            [
                pybamm.grad(v_n),
                pybamm.upwind(v_n),
                pybamm.IndefiniteIntegral(v_n, x_n),
                pybamm.BackwardIndefiniteIntegral(v_n, x_n),
                pybamm.BoundaryValue(v_n, "right"),
                pybamm.BoundaryGradient(v_n, "right"),
                pybamm.SecondaryBroadcast(v_n, "current collector"),
            ],
            [
                pybamm.grad(w_n),
                pybamm.upwind(w_n),
                pybamm.IndefiniteIntegral(w_n, x_n),
                pybamm.BackwardIndefiniteIntegral(w_n, x_n),
                pybamm.BoundaryValue(w_n, "right"),
                pybamm.BoundaryGradient(w_n, "right"),
                pybamm.SecondaryBroadcast(w_n, "current collector"),
            ],
        ):
            self.assertEqual(symbol_v.new_copy(new_children=[w_n]), symbol_w)

        self.assertEqual(
            pybamm.div(pybamm.grad(v_n)).new_copy(new_children=[pybamm.grad(w_n)]),
            pybamm.div(pybamm.grad(w_n)),
        )

        v_s = pybamm.Variable("v", "separator")
        mesh = get_mesh_for_testing()

        for symbol_n, symbol_s in zip(
            [
                pybamm.concatenation(v_n, v_s),
                pybamm.DomainConcatenation([v_n, v_s], mesh),
            ],
            [
                pybamm.concatenation(v_s, v_n),
                pybamm.DomainConcatenation([v_s, v_n], mesh),
            ],
        ):
            self.assertEqual(symbol_n.new_copy(new_children=[v_s, v_n]), symbol_s)

        self.assertEqual(
            pybamm.NumpyConcatenation(a, b, v_s).new_copy(new_children=[b, a, v_n]),
            pybamm.NumpyConcatenation(b, a, v_n),
        )

        v_n_2D = pybamm.Variable(
            "v",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        w_n_2D = pybamm.Variable(
            "w",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        vec = pybamm.Vector([1, 2, 3, 4, 5])
        vec_b = pybamm.Vector([6, 7, 8, 9, 10])
        mat = pybamm.Matrix([[1, 2], [3, 4]])
        mat_b = pybamm.Matrix([[5, 6], [7, 8]])

        self.assertEqual(
            pybamm.TertiaryBroadcast(v_n_2D, "current collector").new_copy(
                new_children=[w_n_2D]
            ),
            pybamm.TertiaryBroadcast(w_n_2D, "current collector"),
        )
        self.assertEqual(
            pybamm.Index(vec, 1).new_copy(new_children=[vec_b]),
            pybamm.Index(vec_b, 1),
        )
        self.assertEqual(
            pybamm.SparseStack(mat, mat).new_copy(new_children=[mat_b, mat_b]),
            pybamm.SparseStack(mat_b, mat_b),
        )

    def test_binary_new_copy_new_children_error(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")

        with self.assertRaisesRegex(ValueError, "must have exactly two children"):
            (a + b).new_copy(new_children=[a])

    def test_unary_new_copy_new_children_error(self):
        vec = pybamm.Vector([1, 2, 3, 4, 5])
        vec_b = pybamm.Vector([6, 7, 8, 9, 10])

        I = pybamm.Index(vec, 1)

        with self.assertRaisesRegex(ValueError, "must have exactly one child"):
            I.new_copy(new_children=[vec, vec_b])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
