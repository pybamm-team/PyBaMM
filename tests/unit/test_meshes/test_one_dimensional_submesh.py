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
        lims = {"a": 1, "b": 2}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Uniform1DSubMesh(lims, None)

    def test_symmetric_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )


class TestExponential1DSubMesh(unittest.TestCase):
    def test_symmetric_mesh_creation_no_parameters_even(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_params = {"side": "symmetric"}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, submesh_params
            )
        }
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )

    def test_symmetric_mesh_creation_no_parameters_odd(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_params = {"side": "symmetric", "stretch": 1.5}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, submesh_params
            )
        }
        var_pts = {r: 21}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )

    def test_left_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_params = {"side": "left"}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, submesh_params
            )
        }
        var_pts = {r: 21}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )

    def test_right_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_params = {"side": "right", "stretch": 2}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, submesh_params
            )
        }
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )


class TestChebyshev1DSubMesh(unittest.TestCase):
    def test_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        submesh_types = {"negative particle": pybamm.Chebyshev1DSubMesh}
        var_pts = {r: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )


class TestUser1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        edges = np.array([0, 0.3, 1])
        submesh_params = {"edges": edges}
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh, submesh_params)

        # error if npts+1 != len(edges)
        lims = {"x_n": {"min": 0, "max": 1}}
        npts = {"x_n": 10}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[0] not equal to edges[0]
        lims = {"x_n": {"min": 0.1, "max": 1}}
        npts = {"x_n": len(edges) - 1}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[-1] not equal to edges[-1]
        lims = {"x_n": {"min": 0, "max": 10}}
        npts = {"x_n": len(edges) - 1}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # no user points
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh)
        with self.assertRaisesRegex(pybamm.GeometryError, "User mesh requires"):
            mesh(None, None)

    def test_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        edges = np.array([0, 0.3, 1])
        submesh_params = {"edges": edges}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.UserSupplied1DSubMesh, submesh_params
            )
        }
        var_pts = {r: len(edges) - 1}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )


class TestSpectralVolume1DSubMesh(unittest.TestCase):
    def test_exceptions(self):
        edges = np.array([0, 0.3, 1])
        submesh_params = {"edges": edges}
        mesh = pybamm.MeshGenerator(pybamm.SpectralVolume1DSubMesh, submesh_params)

        # error if npts+1 != len(edges)
        lims = {"x_n": {"min": 0, "max": 1}}
        npts = {"x_n": 10}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[0] not equal to edges[0]
        lims = {"x_n": {"min": 0.1, "max": 1}}
        npts = {"x_n": len(edges) - 1}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[-1] not equal to edges[-1]
        lims = {"x_n": {"min": 0, "max": 10}}
        npts = {"x_n": len(edges) - 1}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

    def test_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }

        edges = np.array([0, 0.3, 1])
        order = 3
        submesh_params = {"edges": edges, "order": order}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.SpectralVolume1DSubMesh, submesh_params
            )
        }
        var_pts = {r: len(edges) - 1}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].sv_nodes), var_pts[r])
        self.assertEqual(len(mesh["negative particle"].nodes), order * var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )

        # check Chebyshev subdivision locations
        for a, b in zip(
            mesh["negative particle"].edges.tolist(),
            [0, 0.075, 0.225, 0.3, 0.475, 0.825, 1],
        ):
            self.assertAlmostEqual(a, b)

        # test uniform submesh creation
        submesh_params = {"order": order}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.SpectralVolume1DSubMesh, submesh_params
            )
        }
        var_pts = {r: 2}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        for a, b in zip(
            mesh["negative particle"].edges.tolist(),
            [0.0, 0.125, 0.375, 0.5, 0.625, 0.875, 1.0],
        ):
            self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
