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
        discretisation = pybamm.BaseDiscretisation(mesh)
        c = pybamm.Variable("c", domain="x")
        variables = [c]
        y_slices = discretisation.get_variable_slices(variables)
        self.assertEqual(y_slices, {c: slice(0, 100)})
        c_true = mesh.x.points ** 2
        y = c_true
        np.testing.assert_array_equal(y[y_slices[c]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain="x")
        jn = pybamm.Variable("jn", domain="xn")
        variables = [c, d, jn]
        y_slices = discretisation.get_variable_slices(variables)
        self.assertEqual(
            y_slices, {c: slice(0, 100), d: slice(100, 200), jn: slice(200, 240)}
        )
        d_true = 4 * mesh.x.points
        jn_true = mesh.xn.points ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[y_slices[c]], c_true)
        np.testing.assert_array_equal(y[y_slices[d]], d_true)
        np.testing.assert_array_equal(y[y_slices[jn]], jn_true)

    @unittest.skip("")
    def test_discretise_operators(self):
        c = pybamm.Variable("c", domain="x")
        mesh = MeshForTesting()
        discretisation = DiscretisationForTesting(mesh)
        discretised_equation = pybamm.discretise_operators(
            pybamm.grad(c), discretisation
        )
        self.assertTrue(
            isinstance(discretised_equation, pybamm.MatrixVariableMultiplication)
        )
        self.assertTrue(isinstance(discretised_equation.left, pybamm.Matrix))
        self.assertTrue(isinstance(discretised_equation.right, pybamm.Variable))

        pybamm.set_variable_slices([c], mesh)
        y = mesh.x.points ** 2
        np.testing.assert_array_equal(discretised_equation.evaluate(y), c.evaluate(y))

    @unittest.skip("")
    def test_discretise(self):
        c = pybamm.Variable("c", domain="x")
        jn = pybamm.Variable("jn", domain="xn")
        mesh = MeshForTesting()
        discretisation = DiscretisationForTesting(mesh)

        model = {c: pybamm.grad(c), jn: pybamm.div(jn)}
        y = np.concatenate([mesh.x.points ** 2, mesh.xn.points * 3])
        discretised_model = pybamm.discretise(model, discretisation)
        np.testing.assert_array_equal(discretised_model[c].evaluate(y), c.evaluate(y))
        np.testing.assert_array_equal(discretised_model[jn].evaluate(y), jn.evaluate(y))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
