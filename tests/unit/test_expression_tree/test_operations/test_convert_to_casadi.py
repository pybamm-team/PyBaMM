#
# Test for the Simplify class
#
import casadi
import numpy as np
import autograd.numpy as anp
import pybamm
import unittest
from tests import get_mesh_for_testing, get_1p1d_discretisation_for_testing


class TestCasadiConverter(unittest.TestCase):
    def test_convert_scalar_symbols(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        c = pybamm.Scalar(-1)
        d = pybamm.Scalar(2)

        self.assertEqual(a.to_casadi(), casadi.SX(0))
        self.assertEqual(d.to_casadi(), casadi.SX(2))

        # negate
        self.assertEqual((-b).to_casadi(), casadi.SX(-1))
        # absolute value
        self.assertEqual(abs(c).to_casadi(), casadi.SX(1))

        # function
        def sin(x):
            return np.sin(x)

        f = pybamm.Function(sin, b)
        self.assertEqual(f.to_casadi(), casadi.SX(np.sin(1)))

        def myfunction(x, y):
            return x + y

        f = pybamm.Function(myfunction, b, d)
        self.assertEqual(f.to_casadi(), casadi.SX(3))

        # addition
        self.assertEqual((a + b).to_casadi(), casadi.SX(1))
        # subtraction
        self.assertEqual((c - d).to_casadi(), casadi.SX(-3))
        # multiplication
        self.assertEqual((c * d).to_casadi(), casadi.SX(-2))
        # power
        self.assertEqual((c ** d).to_casadi(), casadi.SX(1))
        # division
        self.assertEqual((b / d).to_casadi(), casadi.SX(1 / 2))

    def test_convert_array_symbols(self):
        # Arrays
        a = np.array([1, 2, 3, 4, 5])
        pybamm_a = pybamm.Array(a)
        self.assertTrue(casadi.is_equal(pybamm_a.to_casadi(), casadi.SX(a)))

        casadi_t = casadi.SX.sym("t")
        casadi_y = casadi.SX.sym("y", 10)

        pybamm_t = pybamm.Time()
        pybamm_y = pybamm.StateVector(slice(0, 10))

        # Time
        self.assertEqual(pybamm_t.to_casadi(casadi_t, casadi_y), casadi_t)

        # State Vector
        self.assertTrue(
            casadi.is_equal(pybamm_y.to_casadi(casadi_t, casadi_y), casadi_y)
        )

        # outer product
        outer = pybamm.Outer(pybamm_a, pybamm_a)
        self.assertTrue(casadi.is_equal(outer.to_casadi(), casadi.SX(outer.evaluate())))

    def test_special_functions(self):
        a = np.array([1, 2, 3, 4, 5])
        pybamm_a = pybamm.Array(a)
        self.assertEqual(pybamm.min(pybamm_a).to_casadi(), casadi.SX(1))

    def test_concatenations(self):
        y = np.linspace(0, 1, 10)[:, np.newaxis]
        a = pybamm.Vector(y)
        b = pybamm.Scalar(16)
        c = pybamm.Scalar(3)
        conc = pybamm.NumpyConcatenation(a, b, c)
        self.assertTrue(casadi.is_equal(conc.to_casadi(), casadi.SX(conc.evaluate())))

        # Domain concatenation
        mesh = get_mesh_for_testing()
        a_dom = ["negative electrode"]
        b_dom = ["positive electrode"]
        a = 2 * pybamm.Vector(np.ones_like(mesh[a_dom[0]][0].nodes), domain=a_dom)
        b = pybamm.Vector(np.ones_like(mesh[b_dom[0]][0].nodes), domain=b_dom)
        conc = pybamm.DomainConcatenation([b, a], mesh)
        self.assertTrue(casadi.is_equal(conc.to_casadi(), casadi.SX(conc.evaluate())))

        # 2d
        disc = get_1p1d_discretisation_for_testing()
        a = pybamm.Variable("a", domain=a_dom)
        b = pybamm.Variable("b", domain=b_dom)
        conc = pybamm.Concatenation(a, b)
        disc.set_variable_slices([conc])
        expr = disc.process_symbol(conc)
        y = casadi.SX.sym("y", expr.size)
        x = expr.to_casadi(None, y)
        f = casadi.Function("f", [x], [x])
        y_eval = np.linspace(0, 1, expr.size)
        self.assertTrue(casadi.is_equal(f(y_eval), casadi.SX(expr.evaluate(y=y_eval))))

    def test_convert_differentiated_function(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)

        # function
        def sin(x):
            return anp.sin(x)

        f = pybamm.Function(sin, b).diff(b)
        self.assertEqual(f.to_casadi(), casadi.SX(np.cos(1)))

        def myfunction(x, y):
            return x + y ** 3

        f = pybamm.Function(myfunction, a, b).diff(a)
        self.assertEqual(f.to_casadi(), casadi.SX(1))
        f = pybamm.Function(myfunction, a, b).diff(b)
        self.assertEqual(f.to_casadi(), casadi.SX(3))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
