#
# Tests for the Broadcast classes
#
import pybamm
from tests import get_mesh_for_testing
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
        mesh = get_mesh_for_testing()
        for dom in mesh.keys():
            mesh[dom][0].npts_for_broadcast = mesh[dom][0].npts
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        combined_submeshes = mesh.combine_submeshes(*whole_cell)

        # scalar
        a = pybamm.Scalar(7)
        broad = pybamm.NumpyBroadcast(a, whole_cell, mesh)
        np.testing.assert_array_equal(
            broad.evaluate(), 7 * np.ones_like(combined_submeshes[0].nodes)
        )
        self.assertEqual(broad.domain, whole_cell)

        # state vector
        state_vec = pybamm.StateVector(slice(1, 2))
        broad = pybamm.NumpyBroadcast(state_vec, whole_cell, mesh)
        y = np.linspace(0, 1)
        np.testing.assert_array_equal(
            broad.evaluate(y=y), (y[1:2] * np.ones_like(combined_submeshes[0].nodes))
        )

    @unittest.skip("")
    def test_broadcast_jac(self):
        mesh = get_mesh_for_testing()
        for dom in mesh.keys():
            mesh[dom][0].npts_for_broadcast = mesh[dom][0].npts
        a = pybamm.Scalar(7)
        y = pybamm.StateVector(slice(2, 3))
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh.combine_submeshes(*whole_cell)

        broad_a = pybamm.NumpyBroadcast(a, whole_cell, mesh)
        broad_y = pybamm.NumpyBroadcast(y, whole_cell, mesh)
        broad_y2 = pybamm.NumpyBroadcast(y ** 2, whole_cell, mesh)

        # Create a y vector that is bigger than the mesh
        y_jac = pybamm.StateVector(slice(0, 4))

        y0 = np.ones(4)

        dbroad_a_dyjac = broad_a.jac(y_jac).evaluate()
        dbroad_y_dyjac = broad_y.jac(y_jac).evaluate()
        dbroad_y2_dyjac = broad_y2.jac(y_jac).evaluate(y=y0)
        import ipdb

        ipdb.set_trace()
        np.testing.assert_array_equal()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
