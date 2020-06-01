#
# Test for the scikit-fem Finite Element Mesh class
#
import pybamm
import unittest
import numpy as np


class TestScikitFiniteElement2DSubMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.1,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab width [m]": 0.1,
                "Positive tab centre y-coordinate [m]": 0.3,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, current_collector_dimension=2
        )
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.y: 16, var.z: 24}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
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
        for domain in mesh:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts[var.y] * var_pts[var.z]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_init_failure(self):
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
        }
        geometry = pybamm.battery_geometry(
            include_particles=False, current_collector_dimension=2
        )
        with self.assertRaises(KeyError):
            pybamm.Mesh(geometry, submesh_types, {})

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.y: 10, var.z: 10}
        # there are parameters in the variables that need to be processed
        with self.assertRaises(NotImplementedError):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        lims = {var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitUniform2DSubMesh(lims, None)

        lims = {
            var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            var.x_p: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitUniform2DSubMesh(lims, None)

        lims = {
            var.y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            var.z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        npts = {var.y.id: 10, var.z.id: 10}
        var.z.coord_sys = "not cartesian"
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitUniform2DSubMesh(lims, npts)
        var.z.coord_sys = "cartesian"

    def test_tab_error(self):
        # set variables and submesh types
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 2, var.x_s: 2, var.x_p: 2, var.y: 64, var.z: 64}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
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
            include_particles=False, current_collector_dimension=2
        )
        param.process_geometry(geometry)
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_tab_left_right(self):
        # set variables and submesh types
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 2, var.x_s: 2, var.x_p: 2, var.y: 64, var.z: 64}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
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
            include_particles=False, current_collector_dimension=2
        )
        param.process_geometry(geometry)
        pybamm.Mesh(geometry, submesh_types, var_pts)


class TestScikitFiniteElementChebyshev2DSubMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.1,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab width [m]": 0.1,
                "Positive tab centre y-coordinate [m]": 0.3,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, current_collector_dimension=2
        )
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.y: 16, var.z: 24}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
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
        for domain in mesh:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts[var.y] * var_pts[var.z]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_init_failure(self):
        var = pybamm.standard_spatial_vars

        # only one lim
        lims = {var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitChebyshev2DSubMesh(lims, None)

        # different coord_sys
        lims = {
            var.r_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            var.z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitChebyshev2DSubMesh(lims, None)

        # not y and z
        lims = {
            var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            var.z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitChebyshev2DSubMesh(lims, None)


class TestScikitExponential2DSubMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.1,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab width [m]": 0.1,
                "Positive tab centre y-coordinate [m]": 0.3,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, current_collector_dimension=2
        )
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.y: 16, var.z: 24}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
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
        for domain in mesh:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts[var.y] * var_pts[var.z]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_init_failure(self):
        var = pybamm.standard_spatial_vars

        # only one lim
        lims = {var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitExponential2DSubMesh(lims, None)

        # different coord_sys
        lims = {
            var.r_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            var.z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitExponential2DSubMesh(lims, None)

        # not y and z
        lims = {
            var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            var.z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        with self.assertRaises(pybamm.DomainError):
            pybamm.ScikitExponential2DSubMesh(lims, None)

        # side not top
        with self.assertRaises(pybamm.GeometryError):
            pybamm.ScikitExponential2DSubMesh(None, None, side="bottom")


class TestScikitUser2DSubMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.1,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab width [m]": 0.1,
                "Positive tab centre y-coordinate [m]": 0.3,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, current_collector_dimension=2
        )
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.y: 16, var.z: 24}

        y_edges = np.linspace(0, 0.8, 16)
        z_edges = np.linspace(0, 1, 24)

        submesh_params = {"y_edges": y_edges, "z_edges": z_edges}
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
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
        for domain in mesh:
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts[var.y] * var_pts[var.z]
                self.assertEqual(mesh[domain].npts, npts)
            else:
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_exceptions(self):
        var = pybamm.standard_spatial_vars
        lims = {var.y: {"min": 0, "max": 1}}
        y_edges = np.array([0, 0.3, 1])
        z_edges = np.array([0, 0.3, 1])
        submesh_params = {"y_edges": y_edges, "z_edges": z_edges}
        mesh = pybamm.MeshGenerator(pybamm.UserSupplied2DSubMesh, submesh_params)
        # test not enough lims
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, None)
        lims = {var.y: {"min": 0, "max": 1}, var.z: {"min": 0, "max": 1}}

        # error if len(edges) != npts
        npts = {var.y.id: 10, var.z.id: 3}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[0] not equal to edges[0]
        lims = {var.y: {"min": 0.1, "max": 1}, var.z: {"min": 0, "max": 1}}
        npts = {var.y.id: 3, var.z.id: 3}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if lims[-1] not equal to edges[-1]
        lims = {var.y: {"min": 0, "max": 1}, var.z: {"min": 0, "max": 1.3}}
        npts = {var.y.id: 3, var.z.id: 3}
        with self.assertRaises(pybamm.GeometryError):
            mesh(lims, npts)

        # error if different coordinate system
        lims = {var.y: {"min": 0, "max": 1}, var.r_n: {"min": 0, "max": 1}}
        npts = {var.y.id: 3, var.r_n.id: 3}
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
