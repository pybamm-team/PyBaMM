#
# Test for the Simplify class
#

import casadi
import numpy as np
import pybamm
import unittest
from tests import get_mesh_for_testing, get_1p1d_discretisation_for_testing
from scipy import special


class TestCasadiConverter(unittest.TestCase):
    def assert_casadi_equal(self, a, b, evalf=False):
        if evalf is True:
            self.assertTrue((casadi.evalf(a) - casadi.evalf(b)).is_zero())
        else:
            self.assertTrue((a - b).is_zero())

    def assert_casadi_almost_equal(self, a, b, decimal=7, evalf=False):
        tol = 1.5 * 10 ** (-decimal)
        if evalf is True:
            self.assertTrue(
                (casadi.fabs(casadi.evalf(a) - casadi.evalf(b)) < tol).is_one()
            )
        else:
            self.assertTrue((casadi.fabs(a - b) < tol).is_one())

    def test_convert_scalar_symbols(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        c = pybamm.Scalar(-1)
        d = pybamm.Scalar(2)
        e = pybamm.Scalar(3)
        g = pybamm.Scalar(3.3)

        self.assertEqual(a.to_casadi(), casadi.MX(0))
        self.assertEqual(d.to_casadi(), casadi.MX(2))

        # negate
        self.assertEqual((-b).to_casadi(), casadi.MX(-1))
        # absolute value
        self.assertEqual(abs(c).to_casadi(), casadi.MX(1))
        # floor
        self.assertEqual(pybamm.Floor(g).to_casadi(), casadi.MX(3))
        # ceiling
        self.assertEqual(pybamm.Ceiling(g).to_casadi(), casadi.MX(4))

        # function
        def square_plus_one(x):
            return x**2 + 1

        f = pybamm.Function(square_plus_one, b)
        self.assertEqual(f.to_casadi(), 2)

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

        # modulo
        self.assertEqual(pybamm.Modulo(e, d).to_casadi(), casadi.MX(1))

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

        # test functions with assert_casadi_equal
        for np_fun in [
            np.sqrt,
            np.tanh,
            np.sinh,
            np.exp,
            np.log,
            np.sign,
            np.sin,
            np.cos,
            np.arccosh,
            np.arcsinh,
        ]:
            self.assert_casadi_almost_equal(
                pybamm.Function(np_fun, c).to_casadi(), casadi.MX(np_fun(3)), evalf=True
            )

        # A workaround to fix the tests running on GitHub Actions -
        # casadi.evalf(
        #       pybamm.Function(np_fun, c).to_casadi()
        # ) - casadi.evalf(casadi.MX(np_fun(3)))
        # is not zero, but a small number of the order 10^-15 when np_func is np.cosh
        for np_fun in [np.cosh]:
            self.assert_casadi_almost_equal(
                pybamm.Function(np_fun, c).to_casadi(),
                casadi.MX(np_fun(3)),
                decimal=14,
                evalf=True,
            )

        # test functions with assert_casadi_almost_equal
        for np_fun in [special.erf]:
            self.assert_casadi_almost_equal(
                pybamm.Function(np_fun, c).to_casadi(),
                casadi.MX(np_fun(3)),
                decimal=15,
                evalf=True,
            )

    def test_interpolation(self):
        x = np.linspace(0, 1)
        y = pybamm.StateVector(slice(0, 2))
        casadi_y = casadi.MX.sym("y", 2)
        # linear
        y_test = np.array([0.4, 0.6])
        for interpolator in ["linear", "cubic"]:
            interp = pybamm.Interpolant(x, 2 * x, y, interpolator=interpolator)
            interp_casadi = interp.to_casadi(y=casadi_y)
            f = casadi.Function("f", [casadi_y], [interp_casadi])
            np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))
        # square
        y = pybamm.StateVector(slice(0, 1))
        interp = pybamm.Interpolant(x, x**2, y, interpolator="cubic")
        interp_casadi = interp.to_casadi(y=casadi_y)
        f = casadi.Function("f", [casadi_y], [interp_casadi])
        np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))

        # len(x)=1 but y is 2d
        y = pybamm.StateVector(slice(0, 1))
        casadi_y = casadi.MX.sym("y", 1)
        data = np.tile(2 * x, (10, 1)).T
        y_test = np.array([0.4])
        for interpolator in ["linear", "cubic"]:
            interp = pybamm.Interpolant(x, data, y, interpolator=interpolator)
            interp_casadi = interp.to_casadi(y=casadi_y)
            f = casadi.Function("f", [casadi_y], [interp_casadi])
            np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))

        # error for pchip interpolator
        interp = pybamm.Interpolant(x, data, y, interpolator="pchip")
        with self.assertRaisesRegex(NotImplementedError, "The interpolator"):
            interp_casadi = interp.to_casadi(y=casadi_y)

        # error for not recognized interpolator
        with self.assertRaisesRegex(ValueError, "interpolator"):
            interp = pybamm.Interpolant(x, data, y, interpolator="idonotexist")
            interp_casadi = interp.to_casadi(y=casadi_y)

        # error for converted children count
        y4 = (
            pybamm.StateVector(slice(0, 1)),
            pybamm.StateVector(slice(0, 1)),
            pybamm.StateVector(slice(0, 1)),
            pybamm.StateVector(slice(0, 1)),
        )
        x4_ = [np.linspace(0, 1) for _ in range(4)]
        x4 = np.column_stack(x4_)
        data4 = 2 * x4  # np.tile(2 * x3, (10, 1)).T
        with self.assertRaisesRegex(ValueError, "Invalid dimension of x"):
            interp = pybamm.Interpolant(x4_, data4, y4, interpolator="linear")
            interp_casadi = interp.to_casadi(y=casadi_y)

    def test_interpolation_2d(self):
        x_ = [np.linspace(0, 1), np.linspace(0, 1)]

        X = list(np.meshgrid(*x_))

        x = np.column_stack([el.reshape(-1, 1) for el in X])
        y = (pybamm.StateVector(slice(0, 2)), pybamm.StateVector(slice(0, 2)))
        casadi_y = casadi.MX.sym("y", 2)
        # linear
        y_test = np.array([0.4, 0.6])
        Y = (2 * x).sum(axis=1).reshape(*[len(el) for el in x_])
        for interpolator in ["linear", "cubic"]:
            interp = pybamm.Interpolant(x_, Y, y, interpolator=interpolator)
            interp_casadi = interp.to_casadi(y=casadi_y)
            f = casadi.Function("f", [casadi_y], [interp_casadi])
            np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))
        # square
        y = (pybamm.StateVector(slice(0, 1)), pybamm.StateVector(slice(0, 1)))
        Y = (x**2).sum(axis=1).reshape(*[len(el) for el in x_])
        interp = pybamm.Interpolant(x_, Y, y, interpolator="linear")
        interp_casadi = interp.to_casadi(y=casadi_y)
        f = casadi.Function("f", [casadi_y], [interp_casadi])
        np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))

        # # len(x)=1 but y is 2d
        # y = pybamm.StateVector(slice(0, 1), slice(0, 1))
        # casadi_y = casadi.MX.sym("y", 1)
        # data = np.tile((2 * x).sum(axis=1), (10, 1)).T
        # y_test = np.array([0.4])
        # for interpolator in ["linear"]:
        #     interp = pybamm.Interpolant(x_, data, y, interpolator=interpolator)
        #     interp_casadi = interp.to_casadi(y=casadi_y)
        #     f = casadi.Function("f", [casadi_y], [interp_casadi])
        #     np.testing.assert_array_almost_equal(interp.evaluate(y=y_test), f(y_test))

        # error for pchip interpolator
        with self.assertRaisesRegex(ValueError, "interpolator should be"):
            interp = pybamm.Interpolant(x_, Y, y, interpolator="pchip")
            interp_casadi = interp.to_casadi(y=casadi_y)

    def test_interpolation_3d(self):
        def f(x, y, z):
            return 2 * x**3 + 3 * y**2 - z

        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)
        z = np.linspace(7, 9, 33)
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
        data = f(xg, yg, zg)

        var1 = pybamm.StateVector(slice(0, 1))
        var2 = pybamm.StateVector(slice(1, 2))
        var3 = pybamm.StateVector(slice(2, 3))

        x_in = (x, y, z)
        interp = pybamm.Interpolant(
            x_in, data, (var1, var2, var3), interpolator="linear"
        )

        casadi_y = casadi.MX.sym("y", 3)
        interp_casadi = interp.to_casadi(y=casadi_y)
        casadi_f = casadi.Function("f", [casadi_y], [interp_casadi])

        y_test = np.array([1, 5, 8])

        casadi_sol = casadi_f(y_test)
        true_value = f(1, 5, 8)

        self.assertIsInstance(casadi_sol, casadi.DM)

        np.testing.assert_equal(true_value, casadi_sol.__float__())

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
        conc = pybamm.concatenation(a, b)
        disc.set_variable_slices([conc])
        expr = disc.process_symbol(conc)
        y = casadi.SX.sym("y", expr.size)
        x = expr.to_casadi(None, y)
        f = casadi.Function("f", [x], [x])
        y_eval = np.linspace(0, 1, expr.size)
        self.assert_casadi_equal(f(y_eval), casadi.SX(expr.evaluate(y=y_eval)))

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
