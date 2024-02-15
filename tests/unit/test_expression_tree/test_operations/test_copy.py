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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
