#
# Test for the scikit-fem Finite Element Mesh class
#
from tests import TestCase
import pybamm
import unittest
import numpy as np


def get_param():
    return pybamm.ParameterValues(
        {
            "Electrode width [m]": 0.4,
            "Electrode height [m]": 0.5,
            "Negative tab width [m]": 0.1,
            "Negative tab centre y-coordinate [m]": 0.1,
            "Negative tab centre z-coordinate [m]": 0.5,
            "Positive tab width [m]": 0.1,
            "Positive tab centre y-coordinate [m]": 0.3,
            "Positive tab centre z-coordinate [m]": 0.5,
            "Negative electrode thickness [m]": 0.3,
            "Separator thickness [m]": 0.4,
            "Positive electrode thickness [m]": 0.3,
        }
    )


class TestScikitFiniteElement2DSubMesh(TestCase):
    def test_mesh_creation(self):
        param = get_param()
        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "y": 16, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative electrode"].edges[0], 0)
        self.assertEqual(mesh["positive electrode"].edges[-1], 1)

        # check internal boundary locations
        self.assertEqual(
            mesh["negative electrode"].edges[-1], mesh["separator"].edges[0]
        )
        self.assertEqual(
            mesh["positive electrode"].edges[0], mesh["separator"].edges[-1]
        )
        for domain in mesh.base_domains:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts["y"] * var_pts["z"]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_init_failure(self):
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
        }
        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        with self.assertRaises(KeyError):
            pybamm.Mesh(geometry, submesh_types, {})

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "y": 10, "z": 10}
        # there are parameters in the variables that need to be processed
        with self.assertRaisesRegex(
            pybamm.DiscretisationError,
            "Parameter values have not yet been set for geometry",
        ):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        lims = {"x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitUniform2DSubMesh(lims, None)

        lims = {
            "x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            "x_p": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitUniform2DSubMesh(lims, None)

        lims = {
            "y": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            "z": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        npts = {"y": 10, "z": 10}
        z = pybamm.SpatialVariable("z", domain="not cartesian")
        lims = {
            "y": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitUniform2DSubMesh(lims, npts)

    def test_tab_error(self):
        # set variables and submesh types
        var_pts = {"x_n": 2, "x_s": 2, "x_p": 2, "y": 64, "z": 64}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
        }

        # set base parameters
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.1,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab centre y-coordinate [m]": 10,
                "Positive tab centre z-coordinate [m]": 10,
                "Positive tab width [m]": 0.1,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        # check error raised if tab not on boundary
        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        param.process_geometry(geometry)
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_tab_left_right(self):
        # set variables and submesh types
        var_pts = {"x_n": 2, "x_s": 2, "x_p": 2, "y": 64, "z": 64}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
        }

        # set base parameters
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.0,
                "Negative tab centre z-coordinate [m]": 0.25,
                "Positive tab centre y-coordinate [m]": 0.4,
                "Positive tab centre z-coordinate [m]": 0.25,
                "Positive tab width [m]": 0.1,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        # check mesh can be built
        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        param.process_geometry(geometry)
        pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_to_json(self):
        param = get_param()
        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "y": 16, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        mesh_json = mesh.to_json()

        expected_json = {
            "submesh_pts": {
                "negative electrode": {"x_n": 10},
                "separator": {"x_s": 7},
                "positive electrode": {"x_p": 12},
                "current collector": {"y": 16, "z": 24},
            },
            "base_domains": [
                "negative electrode",
                "separator",
                "positive electrode",
                "current collector",
            ],
        }

        self.assertEqual(mesh_json, expected_json)

        # test Uniform2DSubMesh serialisation

        submesh = mesh["current collector"].to_json()

        expected_submesh = {
            "edges": {
                "y": [
                    0.0,
                    0.02666666666666667,
                    0.05333333333333334,
                    0.08,
                    0.10666666666666667,
                    0.13333333333333333,
                    0.16,
                    0.18666666666666668,
                    0.21333333333333335,
                    0.24000000000000002,
                    0.26666666666666666,
                    0.29333333333333333,
                    0.32,
                    0.3466666666666667,
                    0.37333333333333335,
                    0.4,
                ],
                "z": [
                    0.0,
                    0.021739130434782608,
                    0.043478260869565216,
                    0.06521739130434782,
                    0.08695652173913043,
                    0.10869565217391304,
                    0.13043478260869565,
                    0.15217391304347827,
                    0.17391304347826086,
                    0.19565217391304346,
                    0.21739130434782608,
                    0.2391304347826087,
                    0.2608695652173913,
                    0.2826086956521739,
                    0.30434782608695654,
                    0.32608695652173914,
                    0.34782608695652173,
                    0.3695652173913043,
                    0.3913043478260869,
                    0.41304347826086957,
                    0.43478260869565216,
                    0.45652173913043476,
                    0.4782608695652174,
                    0.5,
                ],
            },
            "coord_sys": "cartesian",
            "tabs": {
                "negative": {"y_centre": 0.1, "z_centre": 0.5, "width": 0.1},
                "positive": {"y_centre": 0.3, "z_centre": 0.5, "width": 0.1},
            },
        }

        self.assertEqual(submesh, expected_submesh)

        new_submesh = pybamm.ScikitUniform2DSubMesh._from_json(submesh)

        for x, y in zip(mesh['current collector'].edges, new_submesh.edges):
            np.testing.assert_array_equal(x, y)


class TestScikitFiniteElementChebyshev2DSubMesh(TestCase):
    def test_mesh_creation(self):
        param = get_param()

        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "y": 16, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(pybamm.ScikitChebyshev2DSubMesh),
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative electrode"].edges[0], 0)
        self.assertEqual(mesh["positive electrode"].edges[-1], 1)

        # check internal boundary locations
        self.assertEqual(
            mesh["negative electrode"].edges[-1], mesh["separator"].edges[0]
        )
        self.assertEqual(
            mesh["positive electrode"].edges[0], mesh["separator"].edges[-1]
        )
        for domain in mesh.base_domains:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts["y"] * var_pts["z"]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_init_failure(self):
        # only one lim
        lims = {"x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitChebyshev2DSubMesh(lims, None)

        # different coord_sys
        lims = {
            "r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            "z": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitChebyshev2DSubMesh(lims, None)

        # not y and z
        lims = {
            "x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            "z": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitChebyshev2DSubMesh(lims, None)


class TestScikitExponential2DSubMesh(TestCase):
    def test_mesh_creation(self):
        param = get_param()

        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "y": 16, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(
                pybamm.ScikitExponential2DSubMesh
            ),
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative electrode"].edges[0], 0)
        self.assertEqual(mesh["positive electrode"].edges[-1], 1)

        # check internal boundary locations
        self.assertEqual(
            mesh["negative electrode"].edges[-1], mesh["separator"].edges[0]
        )
        self.assertEqual(
            mesh["positive electrode"].edges[0], mesh["separator"].edges[-1]
        )
        for domain in mesh.base_domains:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts["y"] * var_pts["z"]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_init_failure(self):
        # only one lim
        lims = {"x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitExponential2DSubMesh(lims, None)

        # different coord_sys
        lims = {
            "r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            "z": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitExponential2DSubMesh(lims, None)

        # not y and z
        lims = {
            "x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            "z": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitExponential2DSubMesh(lims, None)

        # side not top
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitExponential2DSubMesh(None, None, side="bottom")


class TestScikitUser2DSubMesh(TestCase):
    def test_mesh_creation(self):
        param = get_param()

        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 2}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "y": 16, "z": 24}

        y_edges = np.linspace(0, 0.4, 16)
        z_edges = np.linspace(0, 0.5, 24)

        submesh_params = {"y_edges": y_edges, "z_edges": z_edges}
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.MeshGenerator(
                pybamm.UserSupplied2DSubMesh, submesh_params
            ),
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative electrode"].edges[0], 0)
        self.assertEqual(mesh["positive electrode"].edges[-1], 1)

        # check internal boundary locations
        self.assertEqual(
            mesh["negative electrode"].edges[-1], mesh["separator"].edges[0]
        )
        self.assertEqual(
            mesh["positive electrode"].edges[0], mesh["separator"].edges[-1]
        )
        for domain in mesh.base_domains:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts["y"] * var_pts["z"]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_exceptions(self):
        lims = {"y": {"min": 0, "max": 1}}
        y_edges = np.array([0, 0.3, 1])
        z_edges = np.array([0, 0.3, 1])
        submesh_params = {"y_edges": y_edges, "z_edges": z_edges}
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied2DSubMesh, submesh_params)
        # test not enough lims
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, None)
        lims = {"y": {"min": 0, "max": 1}, "z": {"min": 0, "max": 1}}

        # error if len(edges) != npts
        npts = {"y": 10, "z": 3}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[0] not equal to edges[0]
        lims = {"y": {"min": 0.1, "max": 1}, "z": {"min": 0, "max": 1}}
        npts = {"y": 3, "z": 3}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[-1] not equal to edges[-1]
        lims = {"y": {"min": 0, "max": 1}, "z": {"min": 0, "max": 1.3}}
        npts = {"y": 3, "z": 3}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if different coordinate system
        lims = {"y": {"min": 0, "max": 1}, "r_n": {"min": 0, "max": 1}}
        npts = {"y": 3, "r_n": 3}
        with self.assertRaises(pybamm.DomainError):
            mesh(lims, npts)

        mesh = pybamm.MeshGenerator(pybamm.UserSupplied2DSubMesh)
        with self.assertRaisesRegex(pybamm.GeometryError, "User mesh requires"):
            mesh(None, None)

        submesh_params = {"y_edges": np.array([0, 0.3, 1])}
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied2DSubMesh, submesh_params)
        with self.assertRaisesRegex(pybamm.GeometryError, "User mesh requires"):
            mesh(None, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
