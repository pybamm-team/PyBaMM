#
# Tests for the base model class
#
import pybamm

import numpy as np
import unittest


class MeshForTesting(pybamm.BaseMesh):
    def __init__(self):
        super().__init__(None)
        self.set_submesh("whole_cell", np.linspace(0, 1, 100))
        self.set_submesh("negative_electrode", self.whole_cell.nodes[:40])


class DiscretisationForTesting(pybamm.MatrixVectorDiscretisation):
    """Interpolating operators."""

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient_matrix(self, domain):
        n = getattr(self.mesh, domain[0]).npts
        return pybamm.Matrix(np.eye(n))

    def divergence_matrix(self, domain):
        n = getattr(self.mesh, domain[0]).npts
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
        c = pybamm.Variable("c", domain=["whole_cell"])
        variables = [c]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(y_slices, {c.id: slice(0, 100)})
        c_true = mesh.whole_cell.nodes ** 2
        y = c_true
        np.testing.assert_array_equal(y[y_slices[c.id]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain=["whole_cell"])
        jn = pybamm.Variable("jn", domain=["negative_electrode"])
        variables = [c, d, jn]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(
            y_slices,
            {c.id: slice(0, 100), d.id: slice(100, 200), jn.id: slice(200, 240)},
        )
        d_true = 4 * mesh.whole_cell.nodes
        jn_true = mesh.negative_electrode.nodes ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[y_slices[c.id]], c_true)
        np.testing.assert_array_equal(y[y_slices[d.id]], d_true)
        np.testing.assert_array_equal(y[y_slices[jn.id]], jn_true)

    def test_process_symbol_base(self):
        disc = pybamm.BaseDiscretisation(None)

        # variable
        var = pybamm.Variable("var")
        y_slices = {var.id: slice(53)}
        var_disc = disc.process_symbol(var, None, y_slices, None)
        self.assertTrue(isinstance(var_disc, pybamm.StateVector))
        self.assertEqual(var_disc._y_slice, y_slices[var.id])
        # scalar
        scal = pybamm.Scalar(5)
        scal_disc = disc.process_symbol(scal, None, None, None)
        self.assertTrue(isinstance(scal_disc, pybamm.Scalar))
        self.assertEqual(scal_disc.value, scal.value)

        # parameter
        par = pybamm.Parameter("par")
        with self.assertRaises(TypeError):
            disc.process_symbol(par, None, None, None)

        # binary operator
        bin = var + scal
        bin_disc = disc.process_symbol(bin, None, y_slices, None)
        self.assertTrue(isinstance(bin_disc, pybamm.Addition))
        self.assertTrue(isinstance(bin_disc.children[0], pybamm.StateVector))
        self.assertTrue(isinstance(bin_disc.children[1], pybamm.Scalar))

        # non-spatial unary operator
        # TODO: none of these implemented yet

    def test_discretise_spatial_operator(self):
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        var = pybamm.Variable("var", domain=["whole_cell"])
        y_slices = disc.get_variable_slices([var])
        for eqn in [pybamm.grad(var), pybamm.div(var)]:
            eqn_disc = disc.process_symbol(eqn, var.domain, y_slices, {})

            self.assertTrue(isinstance(eqn_disc, pybamm.Multiplication))
            self.assertTrue(isinstance(eqn_disc.children[0], pybamm.Matrix))
            self.assertTrue(isinstance(eqn_disc.children[1], pybamm.StateVector))

            y = mesh.whole_cell.nodes ** 2
            var_disc = disc.process_symbol(var, None, y_slices, None)
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
        c = pybamm.Variable("c", domain=["whole_cell"])
        initial_conditions = {c: pybamm.Scalar(3)}
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        y0 = disc.process_initial_conditions(initial_conditions)
        np.testing.assert_array_equal(y0, 3 * np.ones_like(mesh.whole_cell.nodes))

        # two equations
        T = pybamm.Variable("T", domain=["negative_electrode"])
        initial_conditions = {c: pybamm.Scalar(3), T: pybamm.Scalar(5)}
        y0 = disc.process_initial_conditions(initial_conditions)
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    3 * np.ones_like(mesh.whole_cell.nodes),
                    5 * np.ones_like(mesh.negative_electrode.nodes),
                ]
            ),
        )

    def test_process_rhs(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole_cell"])
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        # can't process boundary conditions with DiscretisationForTesting
        boundary_conditions = {}
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        y = mesh.whole_cell.nodes ** 2
        y_slices = disc.get_variable_slices(rhs.keys())
        dydt = disc.process_rhs(rhs, boundary_conditions, y_slices)
        np.testing.assert_array_equal(y, dydt(None, y))

        # two equations
        T = pybamm.Variable("T", domain=["negative_electrode"])
        q = pybamm.grad(T)
        rhs = {c: pybamm.div(N), T: pybamm.div(q)}
        boundary_conditions = {}

        y = np.concatenate(
            [mesh.whole_cell.nodes ** 2, mesh.negative_electrode.nodes ** 4]
        )
        y_slices = disc.get_variable_slices(rhs.keys())
        dydt = disc.process_rhs(rhs, boundary_conditions, y_slices)
        np.testing.assert_array_equal(y[y_slices[c.id]], dydt(None, y)[y_slices[c.id]])
        np.testing.assert_array_equal(y[y_slices[T.id]], dydt(None, y)[y_slices[T.id]])

    def test_process_model(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole_cell"])
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        initial_conditions = {c: pybamm.Scalar(3)}
        boundary_conditions = {}
        model = ModelForTesting(rhs, initial_conditions, boundary_conditions)
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        y0, dydt = disc.process_model(model)
        np.testing.assert_array_equal(y0, 3 * np.ones_like(mesh.whole_cell.nodes))
        np.testing.assert_array_equal(y0, dydt(None, y0))

        # two equations
        T = pybamm.Variable("T", domain=["negative_electrode"])
        q = pybamm.grad(T)
        rhs = {c: pybamm.div(N), T: pybamm.div(q)}
        initial_conditions = {c: pybamm.Scalar(2), T: pybamm.Scalar(5)}
        boundary_conditions = {}
        model = ModelForTesting(rhs, initial_conditions, boundary_conditions)

        y0, dydt = disc.process_model(model)
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    2 * np.ones_like(mesh.whole_cell.nodes),
                    5 * np.ones_like(mesh.negative_electrode.nodes),
                ]
            ),
        )
        np.testing.assert_array_equal(y0, dydt(None, y0))

    def test_scalar_to_vector(self):
        a = pybamm.Scalar(5)
        mesh = MeshForTesting()
        disc = pybamm.BaseDiscretisation(mesh)
        a_vec = disc.scalar_to_vector(a, ["whole_cell"])
        self.assertEqual(a_vec.evaluate(None)[0], a.value)
        self.assertEqual(a_vec.shape, mesh.whole_cell.nodes.shape)

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
