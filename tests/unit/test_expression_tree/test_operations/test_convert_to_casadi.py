#
# Test for the Simplify class
#
import casadi
import numpy as np
import pybamm
import unittest
from tests import get_mesh_for_testing, get_1p1d_discretisation_for_testing


class TestCasadiConverter(unittest.TestCase):
    def assert_casadi_equal(self, a, b, evalf=False):
        if evalf is True:
            self.assertTrue((casadi.evalf(a) - casadi.evalf(b)).is_zero())
        else:
            self.assertTrue((a - b).is_zero())

    def test_convert_scalar_symbols(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        c = pybamm.Scalar(-1)
        d = pybamm.Scalar(2)

        self.assertEqual(a.to_casadi(), casadi.MX(0))
        self.assertEqual(d.to_casadi(), casadi.MX(2))

        # negate
        self.assertEqual((-b).to_casadi(), casadi.MX(-1))
        # absolute value
        self.assertEqual(abs(c).to_casadi(), casadi.MX(1))

        # function
        def sin(x):
            return np.sin(x)

        f = pybamm.Function(sin, b)
        self.assertEqual(f.to_casadi(), casadi.MX(np.sin(1)))

        def myfunction(x, y):
            return x + y

        f = pybamm.Function(myfunction, b, d)
        self.assertEqual(f.to_casadi(), casadi.MX(3))

        # use classes to avoid simplification
        # addition
        self.assertEqual((pybamm.Addition(a, b)).to_casadi(), casadi.MX(1))
        # subtraction
        self.assertEqual(pybamm.Subtraction(c, d).to_casadi(), casadi.MX(-3))
        # multiplication
        self.assertEqual(pybamm.Multiplication(c, d).to_casadi(), casadi.MX(-2))
        # power
        self.assertEqual(pybamm.Power(c, d).to_casadi(), casadi.MX(1))
        # division
        self.assertEqual(pybamm.Division(b, d).to_casadi(), casadi.MX(1 / 2))

        # minimum and maximum
        self.assertEqual(pybamm.Minimum(a, b).to_casadi(), casadi.MX(0))
        self.assertEqual(pybamm.Maximum(a, b).to_casadi(), casadi.MX(1))

    def test_convert_array_symbols(self):
        # Arrays
        a = np.array([1, 2, 3, 4, 5])
        pybamm_a = pybamm.Array(a)
        self.assert_casadi_equal(pybamm_a.to_casadi(), casadi.MX(a))

        casadi_t = casadi.MX.sym("t")
        casadi_y = casadi.MX.sym("y", 10)
        casadi_y_dot = casadi.MX.sym("y_dot", 10)

        pybamm_t = pybamm.Time()
        pybamm_y = pybamm.StateVector(slice(0, 10))
        pybamm_y_dot = pybamm.StateVectorDot(slice(0, 10))

        # Time
        self.assertEqual(pybamm_t.to_casadi(casadi_t, casadi_y), casadi_t)

        # State Vector
        self.assert_casadi_equal(pybamm_y.to_casadi(casadi_t, casadi_y), casadi_y)

        # State Vector Dot
        self.assert_casadi_equal(
            pybamm_y_dot.to_casadi(casadi_t, casadi_y, casadi_y_dot), casadi_y_dot
        )

    def test_special_functions(self):
        a = pybamm.Array(np.array([1, 2, 3, 4, 5]))
        self.assert_casadi_equal(pybamm.max(a).to_casadi(), casadi.MX(5), evalf=True)
        self.assert_casadi_equal(pybamm.min(a).to_casadi(), casadi.MX(1), evalf=True)
        b = pybamm.Array(np.array([-2]))
        c = pybamm.Array(np.array([3]))
        self.assert_casadi_equal(
            pybamm.Function(np.abs, b).to_casadi(), casadi.MX(2), evalf=True
        )
        self.assert_casadi_equal(
            pybamm.Function(np.abs, c).to_casadi(), casadi.MX(3), evalf=True
        )

    def test_interpolation(self):
        x = np.linspace(0, 1)[:, np.newaxis]
        y = pybamm.StateVector(slice(0, 2))
        casadi_y = casadi.MX.sym("y", 2)
        # linear
        linear = np.hstack([x, 2 * x])
        y_test = np.array([0.4, 0.6])
        for interpolator in ["pchip", "cubic spline"]:
            interp = pybamm.Interpolant(linear, y, interpolator=interpolator)
            interp_casadi = interp.to_casadi(y=casadi_y)
            f = casadi.Function("f", [casadi_y], [interp_casadi])
            np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))
        # square
        square = np.hstack([x, x ** 2])
        y = pybamm.StateVector(slice(0, 1))
        for interpolator in ["pchip", "cubic spline"]:
            interp = pybamm.Interpolant(square, y, interpolator=interpolator)
            interp_casadi = interp.to_casadi(y=casadi_y)
            f = casadi.Function("f", [casadi_y], [interp_casadi])
            np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))

    def test_concatenations(self):
        y = np.linspace(0, 1, 10)[:, np.newaxis]
        a = pybamm.Vector(y)
        b = pybamm.Scalar(16)
        c = pybamm.Scalar(3)
        conc = pybamm.NumpyConcatenation(a, b, c)
        self.assert_casadi_equal(
            conc.to_casadi(), casadi.MX(conc.evaluate()), evalf=True
        )

        # Domain concatenation
        mesh = get_mesh_for_testing()
        a_dom = ["negative electrode"]
        b_dom = ["separator"]
        a = 2 * pybamm.Vector(np.ones_like(mesh[a_dom[0]].nodes), domain=a_dom)
        b = pybamm.Vector(np.ones_like(mesh[b_dom[0]].nodes), domain=b_dom)
        conc = pybamm.DomainConcatenation([b, a], mesh)
        self.assert_casadi_equal(
            conc.to_casadi(), casadi.MX(conc.evaluate()), evalf=True
        )

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
        self.assert_casadi_equal(f(y_eval), casadi.SX(expr.evaluate(y=y_eval)))

    def test_convert_differentiated_function(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)

        def myfunction(x, y):
            return x + y ** 3

        f = pybamm.Function(myfunction, a, b).diff(a)
        self.assert_casadi_equal(f.to_casadi(), casadi.MX(1), evalf=True)
        f = pybamm.Function(myfunction, a, b).diff(b)
        self.assert_casadi_equal(f.to_casadi(), casadi.MX(3), evalf=True)

    def test_convert_input_parameter(self):
        casadi_t = casadi.MX.sym("t")
        casadi_y = casadi.MX.sym("y", 10)
        casadi_ydot = casadi.MX.sym("ydot", 10)
        casadi_inputs = {
            "Input 1": casadi.MX.sym("Input 1"),
            "Input 2": casadi.MX.sym("Input 2"),
        }

        pybamm_y = pybamm.StateVector(slice(0, 10))
        pybamm_u1 = pybamm.InputParameter("Input 1")
        pybamm_u2 = pybamm.InputParameter("Input 2")

        # Input only
        self.assert_casadi_equal(
            pybamm_u1.to_casadi(casadi_t, casadi_y, casadi_ydot, casadi_inputs),
            casadi_inputs["Input 1"],
        )

        # More complex
        expr = pybamm_u1 + pybamm_y
        self.assert_casadi_equal(
            expr.to_casadi(casadi_t, casadi_y, casadi_ydot, casadi_inputs),
            casadi_inputs["Input 1"] + casadi_y,
        )
        expr = pybamm_u2 * pybamm_y
        self.assert_casadi_equal(
            expr.to_casadi(casadi_t, casadi_y, casadi_ydot, casadi_inputs),
            casadi_inputs["Input 2"] * casadi_y,
        )

    def test_convert_external_variable(self):
        casadi_t = casadi.MX.sym("t")
        casadi_y = casadi.MX.sym("y", 10)
        casadi_inputs = {
            "External 1": casadi.MX.sym("External 1", 3),
            "External 2": casadi.MX.sym("External 2", 10),
        }

        pybamm_y = pybamm.StateVector(slice(0, 10))
        pybamm_u1 = pybamm.ExternalVariable("External 1", 3)
        pybamm_u2 = pybamm.ExternalVariable("External 2", 10)

        # External only
        self.assert_casadi_equal(
            pybamm_u1.to_casadi(casadi_t, casadi_y, inputs=casadi_inputs),
            casadi_inputs["External 1"],
        )

        # More complex
        expr = pybamm_u2 + pybamm_y
        self.assert_casadi_equal(
            expr.to_casadi(casadi_t, casadi_y, inputs=casadi_inputs),
            casadi_inputs["External 2"] + casadi_y,
        )

    def test_errors(self):
        y = pybamm.StateVector(slice(0, 10))
        with self.assertRaisesRegex(
            ValueError, "Must provide a 'y' for converting state vectors"
        ):
            y.to_casadi()
        y_dot = pybamm.StateVectorDot(slice(0, 10))
        with self.assertRaisesRegex(
            ValueError, "Must provide a 'y_dot' for converting state vectors"
        ):
            y_dot.to_casadi()
        var = pybamm.Variable("var")
        with self.assertRaisesRegex(TypeError, "Cannot convert symbol of type"):
            var.to_casadi()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
