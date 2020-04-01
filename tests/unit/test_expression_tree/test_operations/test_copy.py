#
# Test for making copies
#
import numpy as np
import pybamm
import unittest
from tests import get_mesh_for_testing


class TestCopy(unittest.TestCase):
    def test_symbol_new_copy(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        v_n = pybamm.Variable("v", "negative electrode")
        v_s = pybamm.Variable("v", "separator")
        vec = pybamm.Vector(np.array([1, 2, 3, 4, 5]))
        mesh = get_mesh_for_testing()

        for symbol in [
            a + b,
            a - b,
            a * b,
            a / b,
            a ** b,
            -a,
            abs(a),
            pybamm.Function(np.sin, a),
            pybamm.FunctionParameter("function", {"a": a}),
            pybamm.grad(a),
            pybamm.div(a),
            pybamm.Integral(a, pybamm.t),
            pybamm.BoundaryValue(v_n, "right"),
            pybamm.BoundaryGradient(v_n, "right"),
            pybamm.PrimaryBroadcast(a, "domain"),
            pybamm.SecondaryBroadcast(v_n, "current collector"),
            pybamm.FullBroadcast(a, "domain", {"secondary": "other domain"}),
            pybamm.Concatenation(v_n, v_s),
            pybamm.NumpyConcatenation(a, b, v_s),
            pybamm.DomainConcatenation([v_n, v_s], mesh),
            pybamm.Parameter("param"),
            pybamm.InputParameter("param"),
            pybamm.StateVector(slice(0, 56)),
            pybamm.Matrix(np.ones((50, 40))),
            pybamm.SpatialVariable("x", ["negative electrode"]),
            pybamm.t,
            pybamm.Index(vec, 1),
        ]:
            self.assertEqual(symbol.id, symbol.new_copy().id)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
