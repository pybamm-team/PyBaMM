#
# Tests for the base model class
#
import pybamm

import numpy as np
import unittest


class MeshForTesting(object):
    def __init__(self):
        self.whole_cell = SubMeshForTesting(np.linspace(0, 1, 100))
        self.negative_electrode = SubMeshForTesting(self.whole_cell.points[:40])


class SubMeshForTesting(object):
    def __init__(self, points):
        self.points = points
        self.npts = points.size


class DiscretisationForTesting(pybamm.MatrixVectorDiscretisation):
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
        self.assertEqual(y_slices, {c: slice(0, 100)})
        c_true = mesh.whole_cell.points ** 2
        y = c_true
        np.testing.assert_array_equal(y[y_slices[c]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain=["whole_cell"])
        jn = pybamm.Variable("jn", domain=["negative_electrode"])
        variables = [c, d, jn]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(
            y_slices, {c: slice(0, 100), d: slice(100, 200), jn: slice(200, 240)}
        )
        d_true = 4 * mesh.whole_cell.points
        jn_true = mesh.negative_electrode.points ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[y_slices[c]], c_true)
        np.testing.assert_array_equal(y[y_slices[d]], d_true)
        np.testing.assert_array_equal(y[y_slices[jn]], jn_true)

    def test_discretise_symbol_base(self):
        disc = pybamm.BaseDiscretisation(None)

        # variable
        var = pybamm.Variable("var")
        y_slices = {var: slice(53)}
        var_disc = disc.discretise_symbol(var, None, y_slices, None)
        self.assertTrue(isinstance(var_disc, pybamm.VariableVector))
        self.assertEqual(var_disc._y_slice, y_slices[var])
        # scalar
        scal = pybamm.Scalar(5)
        scal_disc = disc.discretise_symbol(scal, None, None, None)
        self.assertTrue(isinstance(scal_disc, pybamm.Scalar))
        self.assertEqual(scal_disc.value, scal.value)

        # parameter
        par = pybamm.Parameter("par")
        self.assertRaises(TypeError, disc.discretise_symbol(par, None, None, None))

        # binary operator
        bin = pybamm.BinaryOperator("bin", var, scal)
        bin_disc = disc.discretise_symbol(bin, None, y_slices, None)
        self.assertTrue(isinstance(bin_disc, pybamm.BinaryOperator))
        self.assertTrue(isinstance(bin_disc.left, pybamm.Vector))
        self.assertTrue(isinstance(bin_disc.right, pybamm.Scalar))

        # non-spatial unary operator
        un = pybamm.UnaryOperator("un", var)
        un_disc = disc.discretise_symbol(un, None, y_slices, None)
        self.assertTrue(isinstance(un_disc, pybamm.UnaryOperator))
        self.assertTrue(isinstance(un_disc.child, pybamm.Vector))

    def test_discretise_spatial_operator(self):
        # no boundary conditions
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        var = pybamm.Variable("var", domain=["whole_cell"])
        y_slices = disc.get_variable_slices([var])
        for eqn in [pybamm.grad(var), pybamm.div(var)]:
            eqn_disc = disc.discretise_symbol(eqn, var.domain, y_slices, {})

            self.assertTrue(isinstance(eqn_disc, pybamm.MatrixMultiplication))
            self.assertTrue(isinstance(eqn_disc.left, pybamm.Matrix))
            self.assertTrue(isinstance(eqn_disc.right, pybamm.VariableVector))

            y = mesh.whole_cell.points ** 2
            var_disc = disc.discretise_symbol(var, None, y_slices, None)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(eqn_disc.evaluate(y), var_disc.evaluate(y))

        # with boundary conditions

    @unittest.skip("")
    def test_discretise_model(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole_cell"])
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        initial_conditions = {c: pybamm.Scalar(1)}
        boundary_conditions = {N: (pybamm.Scalar(0), pybamm.Scalar(2))}
        model = ModelForTesting(rhs, initial_conditions, boundary_conditions)
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        y0, dydt = disc.discretise_model(model)
        np.testing.assert_array_equal(y0, np.ones_like(mesh.x.points))
        np.testing.assert_array_equal(y0[1:-1], dydt(y0)[1:-1])
        np.testing.assert_array_equal(dydt(y0)[0], np.array([0]))
        np.testing.assert_array_equal(dydt(y0)[-1], np.array([2]))

        # two equations
        T = pybamm.Variable("T", domain=["negative_electrode"])
        q = pybamm.grad(T)
        rhs[T] = pybamm.div(q)
        initial_conditions[T] = pybamm.Scalar(5)
        boundary_conditions[T] = (pybamm.Scalar(-3), pybamm.Scalar(12))

        y0, dydt = disc.discretise_model(model)

    @unittest.skip("")
    def test_concatenation(self):
        pass


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
