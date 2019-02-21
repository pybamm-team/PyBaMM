#
# Tests for the Broadcast classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests.shared as shared
import unittest
import numpy as np


class TestUnaryOperators(unittest.TestCase):
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
        disc = pybamm.Discretisation(defaults.mesh, defaults.spatial_methods)
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

        # vector
        vec = pybamm.Vector(np.linspace(0, 1))
        broad = pybamm.NumpyBroadcast(vec, whole_cell, mesh)
        np.testing.assert_array_equal(
            broad.evaluate(),
            np.repeat(np.linspace(0, 1)[np.newaxis, :], combined_submeshes.npts, axis=0),
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
