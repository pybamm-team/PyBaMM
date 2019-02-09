#
# Tests for the Unary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests.shared as shared
import unittest
import numpy as np


class TestUnaryOperators(unittest.TestCase):
    def test_unary_operator(self):
        a = pybamm.Symbol("a", domain=["test"])
        un = pybamm.UnaryOperator("unary test", a)
        self.assertEqual(un.children[0].name, a.name)
        self.assertEqual(un.domain, a.domain)

    def test_negation(self):
        a = pybamm.Symbol("a")
        nega = pybamm.Negate(a)
        self.assertEqual(nega.name, "-")
        self.assertEqual(nega.children[0].name, a.name)

        b = pybamm.Scalar(4)
        negb = pybamm.Negate(b)
        self.assertEqual(negb.evaluate(), -4)

    def test_absolute(self):
        a = pybamm.Symbol("a")
        absa = pybamm.AbsoluteValue(a)
        self.assertEqual(absa.name, "abs")
        self.assertEqual(absa.children[0].name, a.name)

        b = pybamm.Scalar(-4)
        absb = pybamm.AbsoluteValue(b)
        self.assertEqual(absb.evaluate(), 4)

    def test_gradient(self):
        a = pybamm.Symbol("a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.children[0].name, a.name)

    def test_printing(self):
        a = pybamm.Symbol("a")
        self.assertEqual(str(-a), "-a")
        grad = pybamm.Gradient(a)
        self.assertEqual(grad.name, "grad")
        self.assertEqual(str(grad), "grad(a)")

    def test_id(self):
        a = pybamm.Scalar(4)
        un1 = pybamm.UnaryOperator("test", a)
        un2 = pybamm.UnaryOperator("test", a)
        un3 = pybamm.UnaryOperator("new test", a)
        self.assertEqual(un1.id, un2.id)
        self.assertNotEqual(un1.id, un3.id)
        a = pybamm.Scalar(4)
        un4 = pybamm.UnaryOperator("test", a)
        self.assertEqual(un1.id, un4.id)
        d = pybamm.Scalar(42)
        un5 = pybamm.UnaryOperator("test", d)
        self.assertNotEqual(un1.id, un5.id)

    def test_broadcast(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.Broadcast(a, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast")
        self.assertEqual(broad_a.children[0].name, a.name)
        self.assertEqual(broad_a.domain, ["negative electrode"])

        b = pybamm.Symbol("b", domain=["negative electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Broadcast(b, ["separator"])

    def test_numpy_broadcast(self):
        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        disc = shared.DiscretisationForTesting(
            defaults.mesh_type, defaults.submesh_pts, defaults.submesh_types
        )
        disc.mesh_geometry(defaults.geometry)
        mesh = disc.mesh

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        combined_submeshes = mesh.combine_submeshes(*whole_cell)
        # scalar
        a = pybamm.Scalar(7)
        broad = pybamm.NumpyBroadcast(a, whole_cell, mesh)
        np.testing.assert_array_equal(
            broad.evaluate(), 7 * np.ones_like(combined_submeshes.nodes)
        )
        self.assertEqual(broad.domain, whole_cell)

        # vector
        vec = pybamm.Vector(np.linspace(0, 1))
        broad = pybamm.NumpyBroadcast(vec, whole_cell, mesh)
        np.testing.assert_array_equal(
            broad.evaluate(),
            np.linspace(0, 1)[:, np.newaxis] * np.ones_like(combined_submeshes.nodes),
        )

        self.assertEqual(broad.domain, whole_cell)

        # state vector
        state_vec = pybamm.StateVector(slice(1, 2))
        broad = pybamm.NumpyBroadcast(state_vec, whole_cell, mesh)
        y = np.vstack([np.linspace(0, 1), np.linspace(0, 2)])
        np.testing.assert_array_equal(
            broad.evaluate(y=y), (y[1:2].T * np.ones_like(combined_submeshes.nodes)).T
        )

        # state vector - bad input
        state_vec = pybamm.StateVector(slice(1, 5))
        broad = pybamm.NumpyBroadcast(state_vec, whole_cell, mesh)
        y = np.vstack([np.linspace(0, 1), np.linspace(0, 2)]).T
        with self.assertRaises(AssertionError):
            broad.evaluate(y=y)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
