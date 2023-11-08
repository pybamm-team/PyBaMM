#
# Test for the Finite Volume Mesh class
#
from tests import TestCase
import pybamm
import numpy as np
import unittest


def get_param():
    return pybamm.ParameterValues(
        {
            "Negative electrode thickness [m]": 0.1,
            "Separator thickness [m]": 0.2,
            "Positive electrode thickness [m]": 0.3,
            "Negative particle radius [m]": 0.4,
            "Positive particle radius [m]": 0.5,
        }
    )


class TestMesh(TestCase):
    def test_mesh_creation_no_parameters(self):
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

        # check geometry
        self.assertEqual(mesh.geometry, geometry)

        # check boundary locations
        self.assertEqual(mesh["negative particle"].edges[0], 0)
        self.assertEqual(mesh["negative particle"].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"].edges),
            len(mesh["negative particle"].nodes) + 1,
        )

        # errors if old format
        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }
        with self.assertRaisesRegex(
            pybamm.GeometryError, "Geometry should no longer be given keys"
        ):
            mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_creation(self):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check geometry
        self.assertEqual(mesh.geometry, geometry)

        # check boundary locations
        self.assertEqual(mesh["negative electrode"].edges[0], 0)
        self.assertAlmostEqual(mesh["positive electrode"].edges[-1], 0.6)

        # check internal boundary locations
        self.assertEqual(
            mesh["negative electrode"].edges[-1], mesh["separator"].edges[0]
        )
        self.assertEqual(
            mesh["positive electrode"].edges[0], mesh["separator"].edges[-1]
        )
        for domain in mesh.base_domains:
            if domain != "current collector":
                self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

    def test_init_failure(self):
        geometry = pybamm.battery_geometry()
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }
        with self.assertRaisesRegex(KeyError, "Points not given"):
            pybamm.Mesh(geometry, submesh_types, {})

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12}
        geometry = pybamm.battery_geometry(options={"dimensionality": 1})
        with self.assertRaisesRegex(KeyError, "Points not given"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Not processing geometry parameters
        geometry = pybamm.battery_geometry()

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

        with self.assertRaisesRegex(pybamm.DiscretisationError, "Parameter values"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

        # Geometry has an unrecognized variable type
        geometry["negative electrode"] = {
            "x_n": {"min": 0, "max": pybamm.Variable("var")}
        }
        with self.assertRaisesRegex(NotImplementedError, "for symbol var"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_sizes(self):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        self.assertEqual(mesh["negative electrode"].npts, var_pts["x_n"])
        self.assertEqual(mesh["separator"].npts, var_pts["x_s"])
        self.assertEqual(mesh["positive electrode"].npts, var_pts["x_p"])

        self.assertEqual(len(mesh["negative electrode"].edges) - 1, var_pts["x_n"])
        self.assertEqual(len(mesh["separator"].edges) - 1, var_pts["x_s"])
        self.assertEqual(len(mesh["positive electrode"].edges) - 1, var_pts["x_p"])

    def test_mesh_sizes_using_standard_spatial_vars(self):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        self.assertEqual(mesh["negative electrode"].npts, var_pts[var.x_n])
        self.assertEqual(mesh["separator"].npts, var_pts[var.x_s])
        self.assertEqual(mesh["positive electrode"].npts, var_pts[var.x_p])

        self.assertEqual(len(mesh["negative electrode"].edges) - 1, var_pts[var.x_n])
        self.assertEqual(len(mesh["separator"].edges) - 1, var_pts[var.x_s])
        self.assertEqual(len(mesh["positive electrode"].edges) - 1, var_pts[var.x_p])

    def test_combine_submeshes(self):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # create submesh
        submesh = mesh[("negative electrode", "separator")]
        self.assertEqual(submesh.edges[0], 0)
        self.assertEqual(submesh.edges[-1], mesh["separator"].edges[-1])
        np.testing.assert_almost_equal(
            submesh.nodes
            - np.concatenate(
                [mesh["negative electrode"].nodes, mesh["separator"].nodes]
            ),
            0,
        )
        self.assertEqual(submesh.internal_boundaries, [0.1])
        with self.assertRaises(pybamm.DomainError):
            mesh.combine_submeshes("negative electrode", "positive electrode")

        # test errors
        geometry = {
            "negative electrode": {"x_n": {"min": 0, "max": 0.5}},
            "negative particle": {"r_n": {"min": 0.5, "max": 1}},
        }
        param.process_geometry(geometry)

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        with self.assertRaisesRegex(pybamm.DomainError, "trying"):
            mesh.combine_submeshes("negative electrode", "negative particle")

        with self.assertRaisesRegex(
            ValueError, "Submesh domains being combined cannot be empty"
        ):
            mesh.combine_submeshes()

    def test_ghost_cells(self):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        # provide mesh properties
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"].edges[1],
            mesh["negative electrode"].edges[0],
        )
        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"].edges[0],
            -mesh["negative electrode"].edges[1],
        )
        np.testing.assert_array_equal(
            mesh["positive electrode_right ghost cell"].edges[0],
            mesh["positive electrode"].edges[-1],
        )

    def test_mesh_coord_sys(self):
        param = get_param()

        geometry = pybamm.battery_geometry()
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 10, "x_p": 12, "r_n": 5, "r_p": 6}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        for submesh in mesh.values():
            if not isinstance(submesh, pybamm.SubMesh0D):
                self.assertTrue(submesh.coord_sys in pybamm.KNOWN_COORD_SYS)

    def test_unimplemented_meshes(self):
        var_pts = {"x_n": 10, "y": 10}
        geometry = {
            "negative electrode": {
                "x_n": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
            }
        }
        submesh_types = {"negative electrode": pybamm.Uniform1DSubMesh}
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_1plus1D_tabs_left_right(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab centre z-coordinate [m]": 0.0,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 1}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # negative tab should be "left"
        self.assertEqual(mesh["current collector"].tabs["negative tab"], "left")

        # positive tab should be "right"
        self.assertEqual(mesh["current collector"].tabs["positive tab"], "right")

    def test_1plus1D_tabs_right_left(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab centre z-coordinate [m]": 0.0,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.battery_geometry(
            include_particles=False, options={"dimensionality": 1}
        )
        param.process_geometry(geometry)

        var_pts = {"x_n": 10, "x_s": 7, "x_p": 12, "z": 24}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # negative tab should be "right"
        self.assertEqual(mesh["current collector"].tabs["negative tab"], "right")

        # positive tab should be "left"
        self.assertEqual(mesh["current collector"].tabs["positive tab"], "left")

    def test_to_json(self):
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

        mesh_json = mesh.to_json()

        expected_json = {
            "submesh_pts": {"negative particle": {"r": 20}},
            "base_domains": ["negative particle"],
        }

        self.assertEqual(mesh_json, expected_json)


class TestMeshGenerator(TestCase):
    def test_init_name(self):
        mesh_generator = pybamm.MeshGenerator(pybamm.SubMesh0D)
        self.assertEqual(mesh_generator.__repr__(), "Generator for SubMesh0D")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
