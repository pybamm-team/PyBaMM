#
# Test for making copies
#

import numpy as np
import pybamm
import unittest
from tests import get_mesh_for_testing


class TestCopy(unittest.TestCase):
    def test_symbol_create_copy(self):
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
            self.assertEqual(symbol, symbol.create_copy())
            self.assertEqual(symbol.print_name, symbol.create_copy().print_name)

    def test_symbol_create_copy_new_children(self):
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
            new_symbol = symbol_ab.create_copy(new_children=[b, a])
            self.assertEqual(new_symbol, symbol_ba)
            self.assertEqual(new_symbol.print_name, symbol_ba.print_name)

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
            new_symbol = symbol_a.create_copy(new_children=[b])
            self.assertEqual(new_symbol, symbol_b)
            self.assertEqual(new_symbol.print_name, symbol_b.print_name)

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
            new_symbol = symbol_v.create_copy(new_children=[w_n])
            self.assertEqual(new_symbol, symbol_w)
            self.assertEqual(new_symbol.print_name, symbol_w.print_name)

        self.assertEqual(
            pybamm.div(pybamm.grad(v_n)).create_copy(new_children=[pybamm.grad(w_n)]),
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
            new_symbol = symbol_n.create_copy(new_children=[v_s, v_n])
            self.assertEqual(new_symbol, symbol_s)
            self.assertEqual(new_symbol.print_name, symbol_s.print_name)

        self.assertEqual(
            pybamm.NumpyConcatenation(a, b, v_s).create_copy(new_children=[b, a, v_n]),
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
            pybamm.TertiaryBroadcast(v_n_2D, "current collector").create_copy(
                new_children=[w_n_2D]
            ),
            pybamm.TertiaryBroadcast(w_n_2D, "current collector"),
        )
        self.assertEqual(
            pybamm.Index(vec, 1).create_copy(new_children=[vec_b]),
            pybamm.Index(vec_b, 1),
        )
        self.assertEqual(
            pybamm.SparseStack(mat, mat).create_copy(new_children=[mat_b, mat_b]),
            pybamm.SparseStack(mat_b, mat_b),
        )

    def test_create_copy_new_children_binary_error(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")

        with self.assertRaisesRegex(ValueError, "must have exactly two children"):
            (a + b).create_copy(new_children=[a])

    def test_create_copy_new_children_scalars(self):
        a = pybamm.Scalar(2)
        b = pybamm.Scalar(5)

        self.assertEqual((a + b).create_copy(), a + b)
        # a+b produces a scalar, not an addition object.
        with self.assertRaisesRegex(
            ValueError, "Cannot create a copy of a scalar with new children"
        ):
            (a + b).create_copy(new_children=[a, b])

        self.assertEqual(pybamm.Addition(a, b).create_copy(), pybamm.Scalar(7))
        self.assertEqual(
            pybamm.Addition(a, b).create_copy(perform_simplifications=False),
            pybamm.Addition(a, b),
        )

        c = pybamm.Scalar(4)
        d = pybamm.Scalar(8)

        self.assertEqual(
            pybamm.Addition(a, b).create_copy(new_children=[c, d]), pybamm.Scalar(12)
        )
        self.assertEqual(
            pybamm.Addition(a, b).create_copy(
                new_children=[c, d], perform_simplifications=False
            ),
            pybamm.Addition(c, d),
        )

    def test_create_copy_new_children_unary_error(self):
        vec = pybamm.Vector([1, 2, 3, 4, 5])
        vec_b = pybamm.Vector([6, 7, 8, 9, 10])

        I = pybamm.Index(vec, 1)

        with self.assertRaisesRegex(ValueError, "must have exactly one child"):
            I.create_copy(new_children=[vec, vec_b])

    def test_unary_create_copy_no_simplification(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")

        for symbol_a, symbol_b in zip(
            [
                pybamm.Negate(a),
                pybamm.AbsoluteValue(a),
                pybamm.sign(a),
                # boundaryvalue
            ],
            [
                pybamm.Negate(b),
                pybamm.AbsoluteValue(b),
                pybamm.Sign(b),
            ],
        ):
            self.assertEqual(
                symbol_a.create_copy(new_children=[b], perform_simplifications=False),
                symbol_b,
            )

        v_n = pybamm.Variable("v", "negative electrode")
        w_n = pybamm.Variable("w", "negative electrode")

        self.assertEqual(
            pybamm.grad(v_n).create_copy(
                new_children=[w_n], perform_simplifications=False
            ),
            pybamm.Gradient(w_n),
        )

        self.assertEqual(
            pybamm.div(pybamm.grad(v_n)).create_copy(
                new_children=[pybamm.grad(w_n)], perform_simplifications=False
            ),
            pybamm.Divergence(pybamm.grad(w_n)),
        )

        var = pybamm.Variable("var", domain="test")
        ible = pybamm.Variable("ible", domain="test")
        left_extrap = pybamm.BoundaryValue(var, "left")

        self.assertEqual(
            left_extrap.create_copy(new_children=[ible], perform_simplifications=False),
            pybamm.BoundaryValue(ible, "left"),
        )

    def test_unary_create_copy_no_simplification_averages(self):
        a_v = pybamm.Variable("a", domain=["negative electrode"])
        c = pybamm.Variable("a", domain=["current collector"])

        for average, var in zip(
            [
                pybamm.XAverage,
                pybamm.RAverage,
                pybamm.ZAverage,
                pybamm.YZAverage,
            ],
            [a_v, a_v, c, c],
        ):
            self.assertEqual(
                average(var).create_copy(
                    new_children=[var], perform_simplifications=False
                ),
                average(var),
            )

        d = pybamm.Symbol("d", domain=["negative particle size"])
        R = pybamm.SpatialVariable("R", ["negative particle size"])
        geo = pybamm.geometric_parameters
        f_a_dist = geo.n.prim.f_a_dist(R)

        s_a = pybamm.SizeAverage(d, f_a_dist=f_a_dist)

        self.assertEqual(
            s_a.create_copy(new_children=[d], perform_simplifications=False),
            pybamm.SizeAverage(d, f_a_dist=f_a_dist),
        )

    def test_concatenation_create_copy_no_simplification(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        v_n = pybamm.Variable("v", "negative electrode")
        v_s = pybamm.Variable("v", "separator")
        mesh = get_mesh_for_testing()

        for symbol_n, symbol_s in zip(
            [
                pybamm.concatenation(v_n, v_s),
                pybamm.DomainConcatenation([v_n, v_s], mesh),
            ],
            [
                pybamm.ConcatenationVariable(v_s, v_n),
                pybamm.DomainConcatenation([v_s, v_n], mesh),
            ],
        ):
            self.assertEqual(
                symbol_n.create_copy(
                    new_children=[v_s, v_n], perform_simplifications=False
                ),
                symbol_s,
            )

        with self.assertRaisesRegex(
            NotImplementedError, "should always be copied using simplification checks"
        ):
            pybamm.NumpyConcatenation(a, b, v_s).create_copy(
                new_children=[a, b], perform_simplifications=False
            )

    def test_function_create_copy_no_simplification(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")

        self.assertEqual(
            pybamm.Function(np.sin, a).create_copy(
                new_children=[b], perform_simplifications=False
            ),
            pybamm.Function(np.sin, b),
        )

    def test_symbol_new_copy_warning(self):
        with self.assertWarns(DeprecationWarning):
            pybamm.Symbol("a").new_copy()

    def test_symbol_copy_tree(self):
        model = pybamm.lithium_ion.DFN()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        y = model.concatenated_initial_conditions.evaluate()
        copied_rhs = model.concatenated_rhs.create_copy()
        np.testing.assert_array_equal(
            model.concatenated_rhs.evaluate(None, y), copied_rhs.evaluate(None, y)
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
