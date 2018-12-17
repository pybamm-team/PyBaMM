#
# Tests for the base model class
#
import pybamm

import numpy as np
import unittest


class MeshForTesting(object):
    def __init__(self):
        self.x = SubMeshForTesting(np.linspace(0, 1, 100))
        self.xn = SubMeshForTesting(self.x.points[:40])


class SubMeshForTesting(object):
    def __init__(self, points):
        self.points = points
        self.npts = points.size


class DiscretisationForTesting(pybamm.MatrixVectorDiscretisation):
    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient_matrix(self, domain):
        n = getattr(self.mesh, domain).npts
        return pybamm.Matrix(np.eye(n))

    def divergence_matrix(self, domain):
        n = getattr(self.mesh, domain).npts
        return pybamm.Matrix(np.eye(n))


def ModelForTesting(object):
    def __init__(self, rhs, initial_conditions, boundary_conditions):
        self.rhs = rhs
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions


class TestDiscretise(unittest.TestCase):
    def test_discretise_slicing(self):
        # One variable
        mesh = MeshForTesting()
        disc = pybamm.BaseDiscretisation(mesh)
        c = pybamm.Variable("c", domain="x")
        variables = [c]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(y_slices, {c: slice(0, 100)})
        c_true = mesh.x.points ** 2
        y = c_true
        np.testing.assert_array_equal(y[y_slices[c]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain="x")
        jn = pybamm.Variable("jn", domain="xn")
        variables = [c, d, jn]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(
            y_slices, {c: slice(0, 100), d: slice(100, 200), jn: slice(200, 240)}
        )
        d_true = 4 * mesh.x.points
        jn_true = mesh.xn.points ** 3
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
        self.assertTrue(isinstance(var_disc, pybamm.Vector))
        self.assertEqual(var_disc._y_slice, y_slices[var])
        # scalar
        sca = pybamm.Scalar(5)
        sca_disc = disc.discretise_symbol(sca, None, None, None)
        self.assertTrue(isinstance(sca_disc, pybamm.Scalar))
        self.assertEqual(sca_disc.value, sca.value)

        # parameter
        par = pybamm.Parameter("par")
        with self.assertRaises(TypeError):
            disc.discretise_symbol(par, None, None, None)

        # binary operator
        bin = pybamm.BinaryOperator("bin", var, sca)
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
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        var = pybamm.Variable("var", domain="x")
        y_slices = disc.get_variable_slices([var])
        for eqn in [pybamm.grad(var), pybamm.div(var)]:
            eqn_disc = disc.discretise_symbol(eqn, var.domain, y_slices, {})

            self.assertTrue(isinstance(eqn_disc, pybamm.MatrixVectorMultiplication))
            self.assertTrue(isinstance(eqn_disc.left, pybamm.Matrix))
            self.assertTrue(isinstance(eqn_disc.right, pybamm.Vector))

            y = mesh.x.points ** 2
            var_disc = disc.discretise_symbol(var, None, y_slices, None)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(eqn_disc.evaluate(y), var_disc.evaluate(y))

    @unittest.skip("")
    def test_discretise(self):
        c = pybamm.Variable("c", domain="x")
        jn = pybamm.Variable("jn", domain="xn")
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        model = {c: pybamm.grad(c), jn: pybamm.div(jn)}
        y = np.concatenate([mesh.x.points ** 2, mesh.xn.points * 3])
        discretised_model = pybamm.discretise(model, disc)
        np.testing.assert_array_equal(discretised_model[c].evaluate(y), c.evaluate(y))
        np.testing.assert_array_equal(discretised_model[jn].evaluate(y), jn.evaluate(y))

    def test_concatenation(self):
        pass


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
