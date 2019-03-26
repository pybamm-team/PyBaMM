#
# Tests for the Broadcast classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from tests import get_discretisation_for_testing
import unittest
import numpy as np


class TestBroadcasts(unittest.TestCase):
    def test_broadcast(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.Broadcast(a, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast")
        self.assertEqual(broad_a.children[0].name, a.name)
        self.assertEqual(broad_a.domain, ["negative electrode"])

        b = pybamm.Symbol("b", domain=["negative electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Broadcast(b, ["separator"])

    def test_broadcast_number(self):
        broad_a = pybamm.Broadcast(1, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast")
        self.assertIsInstance(broad_a.children[0], pybamm.Symbol)
        self.assertEqual(broad_a.children[0].name, str(1.0))
        self.assertEqual(broad_a.domain, ["negative electrode"])

        b = pybamm.Symbol("b", domain=["negative electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Broadcast(b, ["separator"])

    def test_numpy_broadcast(self):
        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast = mesh[dom].npts
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        combined_submeshes = mesh.combine_submeshes(*whole_cell)

        # scalar
        a = pybamm.Scalar(7)
        broad = pybamm.NumpyBroadcast(a, whole_cell, mesh)
        np.testing.assert_array_equal(
            broad.evaluate(), 7 * np.ones_like(combined_submeshes.nodes)
        )
        self.assertEqual(broad.domain, whole_cell)

        # time
        t = 3 * pybamm.t + 4
        broad = pybamm.NumpyBroadcast(t, whole_cell, mesh)
        self.assertEqual(broad.domain, whole_cell)
        np.testing.assert_array_equal(
            broad.evaluate(t=3), 13 * np.ones_like(combined_submeshes.nodes)
        )
        np.testing.assert_array_equal(
            broad.evaluate(t=np.linspace(0, 1)),
            (
                (3 * np.linspace(0, 1) + 4)[:, np.newaxis]
                * np.ones_like(combined_submeshes.nodes)
            ).T,
        )

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
        with self.assertRaisesRegex(ValueError, "cannot broadcast child with shape"):
            broad.evaluate(y=y)

        # vector - not accepted
        vec = pybamm.Vector(np.ones(5))
        with self.assertRaisesRegex(
            TypeError, "cannot Broadcast a constant Vector or Matrix"
        ):
            broad = pybamm.NumpyBroadcast(vec, whole_cell, mesh)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
