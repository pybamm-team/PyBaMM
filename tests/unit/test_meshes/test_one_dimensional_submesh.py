import pybamm
import unittest
import numpy as np


class TestSubMesh1D(unittest.TestCase):
    def test_tabs(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0}, "positive": {"z_centre": 1}}
        mesh = pybamm.SubMesh1D(edges, None, tabs=tabs)
        self.assertEqual(mesh.tabs["negative tab"], "left")
        self.assertEqual(mesh.tabs["positive tab"], "right")

    def test_exceptions(self):
        edges = np.linspace(0, 1, 10)
        tabs = {"negative": {"z_centre": 0.2}, "positive": {"z_centre": 1}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.SubMesh1D(edges, None, tabs=tabs)


class TestUniform1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        lims = [[0, 1], [0, 1]]
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Uniform1DSubMesh(lims, None)


class TestExponential1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        lims = [[0, 1], [0, 1]]
        mesh = pybamm.GetExponential1DSubMesh()
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, None)

    def test_symmetric_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }

        submesh_types = {
            "negative particle": pybamm.GetExponential1DSubMesh(side="symmetric")
        }
        var_pts = {r: 20}
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

    def test_left_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }

        submesh_types = {
            "negative particle": pybamm.GetExponential1DSubMesh(side="left")
        }
        var_pts = {r: 20}
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

    def test_right_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }

        submesh_types = {
            "negative particle": pybamm.GetExponential1DSubMesh(side="right")
        }
        var_pts = {r: 20}
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


class TestChebyshev1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        lims = [[0, 1], [0, 1]]
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Chebyshev1DSubMesh(lims, None)

    def test_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }

        submesh_types = {"negative particle": pybamm.Chebyshev1DSubMesh}
        var_pts = {r: 20}
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


class TestUser1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        lims = [[0, 1], [0, 1]]
        edges = np.array([0, 0.3, 1])
        mesh = pybamm.GetUserSupplied1DSubMesh(edges)
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
        submesh_types = {"negative particle": pybamm.GetUserSupplied1DSubMesh(edges)}
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
