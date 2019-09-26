import pybamm
import unittest
import numpy as np


class TestUser1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        lims = [[0, 1], [0, 1]]
        edges = np.array([0, 0.3, 1])
        mesh = pybamm.one_dimensional_meshes.GetUserSupplied1DSubMesh(edges)
        # test too many lims
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, None)
        lims = [0, 1]

        # error if len(edges) != npts+1
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, 5)

        # error if lims[0] not equal to edges[0]
        lims = [0.1, 1]
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, len(edges) - 1)

        # error if lims[-1] not equal to edges[-1]
        lims = [0, 0.9]
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, len(edges) - 1)

    def test_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }

        edges = np.array([0, 0.3, 1])
        submesh_types = {
            "negative particle": pybamm.one_dimensional_meshes.GetUserSupplied1DSubMesh(
                edges
            )
        }
        var_pts = {r: len(edges) - 1}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"][0].edges[0], 0)
        self.assertEqual(mesh["negative particle"][0].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"][0].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"][0].edges),
            len(mesh["negative particle"][0].nodes) + 1,
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
