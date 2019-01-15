#
# Tests for the base model class
#
import pybamm

import numpy as np
import unittest


class MeshForTesting(pybamm.BaseMesh):
    def __init__(self):
        super().__init__(None)
        self["whole cell"] = self.submeshclass(np.linspace(0, 1, 100))
        self["negative electrode"] = self.submeshclass(self["whole cell"].nodes[:40])


class DiscretisationForTesting(pybamm.MatrixVectorDiscretisation):
    """Interpolating operators."""

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient_matrix(self, domain):
        n = self.mesh[domain[0]].npts
        return pybamm.Matrix(np.eye(n))

    def divergence_matrix(self, domain):
        n = self.mesh[domain[0]].npts
        return pybamm.Matrix(np.eye(n))


class ModelForTesting(object):
    def __init__(self, rhs, initial_conditions, boundary_conditions):
        self.rhs = rhs
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions


class TestDiscretise(unittest.TestCase):
    def test_discretise_slicing(self):
        # One variable
        mesh = MeshForTesting()
        disc = pybamm.BaseDiscretisation(mesh)
        c = pybamm.Variable("c", domain=["whole cell"])
        variables = [c]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(y_slices, {c.id: slice(0, 100)})
        c_true = mesh["whole cell"].nodes ** 2
        y = c_true
        np.testing.assert_array_equal(y[y_slices[c.id]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain=["whole cell"])
        jn = pybamm.Variable("jn", domain=["negative electrode"])
        variables = [c, d, jn]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(
            y_slices,
            {c.id: slice(0, 100), d.id: slice(100, 200), jn.id: slice(200, 240)},
        )
        d_true = 4 * mesh["whole cell"].nodes
        jn_true = mesh["negative electrode"].nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[y_slices[c.id]], c_true)
        np.testing.assert_array_equal(y[y_slices[d.id]], d_true)
        np.testing.assert_array_equal(y[y_slices[jn.id]], jn_true)

    def test_process_symbol_base(self):
        disc = pybamm.BaseDiscretisation(None)

        # variable
        var = pybamm.Variable("var")
        y_slices = {var.id: slice(53)}
        var_disc = disc.process_symbol(var, None, y_slices)
        self.assertTrue(isinstance(var_disc, pybamm.StateVector))
        self.assertEqual(var_disc._y_slice, y_slices[var.id])
        # scalar
        scal = pybamm.Scalar(5)
        scal_disc = disc.process_symbol(scal, None)
        self.assertTrue(isinstance(scal_disc, pybamm.Scalar))
        self.assertEqual(scal_disc.value, scal.value)

        # parameter
        par = pybamm.Parameter("par")
        par_disc = disc.process_symbol(par, None)
        self.assertTrue(isinstance(par_disc, pybamm.Parameter))
        self.assertEqual(par_disc.name, par.name)

        # binary operator
        bin = var + scal
        bin_disc = disc.process_symbol(bin, None, y_slices)
        self.assertTrue(isinstance(bin_disc, pybamm.Addition))
        self.assertTrue(isinstance(bin_disc.children[0], pybamm.StateVector))
        self.assertTrue(isinstance(bin_disc.children[1], pybamm.Scalar))

        # non-spatial unary operator
        un1 = -var
        un1_disc = disc.process_symbol(un1, None, y_slices)
        self.assertTrue(isinstance(un1_disc, pybamm.Negate))
        self.assertTrue(isinstance(un1_disc.children[0], pybamm.StateVector))

        un2 = abs(scal)
        un2_disc = disc.process_symbol(un2, None)
        self.assertTrue(isinstance(un2_disc, pybamm.AbsoluteValue))
        self.assertTrue(isinstance(un2_disc.children[0], pybamm.Scalar))

    def test_process_complex_expression(self):
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        par1 = pybamm.Parameter("par1")
        par2 = pybamm.Parameter("par2")
        scal1 = pybamm.Scalar("scal1")
        scal2 = pybamm.Scalar("scal2")
        expression = (scal1 * (par1 + var2)) / ((var1 - par2) + scal2)

        disc = pybamm.BaseDiscretisation(None)
        y_slices = {var1.id: slice(53), var2.id: slice(53, 59)}
        exp_disc = disc.process_symbol(expression, None, y_slices)
        self.assertTrue(isinstance(exp_disc, pybamm.Division))
        # left side
        self.assertTrue(isinstance(exp_disc.children[0], pybamm.Multiplication))
        self.assertTrue(isinstance(exp_disc.children[0].children[0], pybamm.Scalar))
        self.assertTrue(isinstance(exp_disc.children[0].children[1], pybamm.Addition))
        self.assertTrue(
            isinstance(exp_disc.children[0].children[1].children[0], pybamm.Parameter)
        )
        self.assertTrue(
            isinstance(exp_disc.children[0].children[1].children[1], pybamm.StateVector)
        )
        self.assertEqual(
            exp_disc.children[0].children[1].children[1].y_slice, y_slices[var2.id]
        )
        # right side
        self.assertTrue(isinstance(exp_disc.children[1], pybamm.Addition))
        self.assertTrue(
            isinstance(exp_disc.children[1].children[0], pybamm.Subtraction)
        )
        self.assertTrue(
            isinstance(exp_disc.children[1].children[0].children[0], pybamm.StateVector)
        )
        self.assertEqual(
            exp_disc.children[1].children[0].children[0].y_slice, y_slices[var1.id]
        )
        self.assertTrue(
            isinstance(exp_disc.children[1].children[0].children[1], pybamm.Parameter)
        )
        self.assertTrue(isinstance(exp_disc.children[1].children[1], pybamm.Scalar))

    def test_discretise_spatial_operator(self):
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        var = pybamm.Variable("var", domain=["whole cell"])
        y_slices = disc.get_variable_slices([var])
        for eqn in [pybamm.grad(var), pybamm.div(var)]:
            eqn_disc = disc.process_symbol(eqn, var.domain, y_slices, {})

            self.assertTrue(isinstance(eqn_disc, pybamm.Multiplication))
            self.assertTrue(isinstance(eqn_disc.children[0], pybamm.Matrix))
            self.assertTrue(isinstance(eqn_disc.children[1], pybamm.StateVector))

            y = mesh["whole cell"].nodes ** 2
            var_disc = disc.process_symbol(var, None, y_slices)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(
                eqn_disc.evaluate(None, y), var_disc.evaluate(None, y)
            )

    def test_core_NotImplementedErrors(self):
        disc = pybamm.BaseDiscretisation(None)
        with self.assertRaises(NotImplementedError):
            disc.gradient(None, None, None, {})
        with self.assertRaises(NotImplementedError):
            disc.divergence(None, None, None, {})
        disc = pybamm.MatrixVectorDiscretisation(None)
        with self.assertRaises(NotImplementedError):
            disc.gradient_matrix(None)
        with self.assertRaises(NotImplementedError):
            disc.divergence_matrix(None)

    def test_process_initial_conditions(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole cell"])
        initial_conditions = {c: pybamm.Scalar(3)}
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        y0 = disc.process_initial_conditions(initial_conditions)
        np.testing.assert_array_equal(y0, 3 * np.ones_like(mesh["whole cell"].nodes))

        # two equations
        T = pybamm.Variable("T", domain=["negative electrode"])
        initial_conditions = {c: pybamm.Scalar(3), T: pybamm.Scalar(5)}
        y0 = disc.process_initial_conditions(initial_conditions)
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    3 * np.ones_like(mesh["whole cell"].nodes),
                    5 * np.ones_like(mesh["negative electrode"].nodes),
                ]
            ),
        )

    def test_process_rhs(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole cell"])
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        # can't process boundary conditions with DiscretisationForTesting
        boundary_conditions = {}
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        y = mesh["whole cell"].nodes ** 2
        y_slices = disc.get_variable_slices(rhs.keys())
        concatenated_rhs = disc.process_rhs(rhs, boundary_conditions, y_slices)
        np.testing.assert_array_equal(y, concatenated_rhs.evaluate(None, y))

        # two equations
        T = pybamm.Variable("T", domain=["negative electrode"])
        q = pybamm.grad(T)
        rhs = {c: pybamm.div(N), T: pybamm.div(q)}
        boundary_conditions = {}

        y = np.concatenate(
            [mesh["whole cell"].nodes ** 2, mesh["negative electrode"].nodes ** 4]
        )
        y_slices = disc.get_variable_slices(rhs.keys())
        concatenated_rhs = disc.process_rhs(rhs, boundary_conditions, y_slices)
        np.testing.assert_array_equal(
            y[y_slices[c.id]], concatenated_rhs.evaluate(None, y)[y_slices[c.id]]
        )
        np.testing.assert_array_equal(
            y[y_slices[T.id]], concatenated_rhs.evaluate(None, y)[y_slices[T.id]]
        )

    def test_process_model(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole cell"])
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        initial_conditions = {c: pybamm.Scalar(3)}
        boundary_conditions = {}
        model = ModelForTesting(rhs, initial_conditions, boundary_conditions)
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        disc.process_model(model)
        y0 = model.initial_conditions
        np.testing.assert_array_equal(y0, 3 * np.ones_like(mesh["whole cell"].nodes))
        np.testing.assert_array_equal(y0, model.rhs.evaluate(None, y0))

        # two equations
        T = pybamm.Variable("T", domain=["negative electrode"])
        q = pybamm.grad(T)
        rhs = {c: pybamm.div(N), T: pybamm.div(q)}
        initial_conditions = {c: pybamm.Scalar(2), T: pybamm.Scalar(5)}
        boundary_conditions = {}
        model = ModelForTesting(rhs, initial_conditions, boundary_conditions)

        disc.process_model(model)
        y0 = model.initial_conditions
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    2 * np.ones_like(mesh["whole cell"].nodes),
                    5 * np.ones_like(mesh["negative electrode"].nodes),
                ]
            ),
        )
        print(model.rhs.evaluate(None, y0))
        np.testing.assert_array_equal(y0, model.rhs.evaluate(None, y0))

    def test_vector_of_ones(self):
        mesh = MeshForTesting()
        disc = pybamm.BaseDiscretisation(mesh)
        vec = disc.vector_of_ones(["whole cell"])
        self.assertEqual(vec.evaluate(None)[0], 1)
        self.assertEqual(vec.shape, mesh["whole cell"].nodes.shape)

    def test_concatenation(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        disc = pybamm.BaseDiscretisation(None)
        conc = disc.concatenate(a, b, c)
        self.assertTrue(isinstance(conc, pybamm.Concatenation))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
