#
# Test for the Finite Volume Mesh class
#
import pybamm
import numpy as np
import unittest


class TestMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width [m]": 0.1,
                "Separator width [m]": 0.2,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative electrode"][0].edges[0], 0)
        self.assertEqual(mesh["positive electrode"][0].edges[-1], 1)

        # check internal boundary locations
        self.assertEqual(
            mesh["negative electrode"][0].edges[-1], mesh["separator"][0].edges[0]
        )
        self.assertEqual(
            mesh["positive electrode"][0].edges[0], mesh["separator"][0].edges[-1]
        )
        for domain in mesh:
            self.assertEqual(len(mesh[domain][0].edges), len(mesh[domain][0].nodes) + 1)

    def test_init_failure(self):
        geometry = pybamm.Geometry1DMacro()
        with self.assertRaises(KeyError):
            pybamm.Mesh(geometry, None, {})

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12}
        geometry = pybamm.Geometry1p1DMicro()
        with self.assertRaises(KeyError):
            pybamm.Mesh(geometry, None, var_pts)

    def test_mesh_sizes(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width [m]": 0.1,
                "Separator width [m]": 0.2,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
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
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        var_id_pts = {var.id: pts for var, pts in var_pts.items()}

        self.assertEqual(mesh["negative electrode"][0].npts, var_id_pts[var.x_n.id])
        self.assertEqual(mesh["separator"][0].npts, var_id_pts[var.x_s.id])
        self.assertEqual(mesh["positive electrode"][0].npts, var_id_pts[var.x_p.id])

        self.assertEqual(
            len(mesh["negative electrode"][0].edges) - 1, var_id_pts[var.x_n.id]
        )
        self.assertEqual(len(mesh["separator"][0].edges) - 1, var_id_pts[var.x_s.id])
        self.assertEqual(
            len(mesh["positive electrode"][0].edges) - 1, var_id_pts[var.x_p.id]
        )

    def test_combine_submeshes(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width [m]": 0.1,
                "Separator width [m]": 0.2,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
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
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        # create submesh
        submesh = mesh.combine_submeshes("negative electrode", "separator")
        self.assertEqual(submesh[0].edges[0], 0)
        self.assertEqual(submesh[0].edges[-1], mesh["separator"][0].edges[-1])
        np.testing.assert_almost_equal(
            submesh[0].nodes
            - np.concatenate(
                [mesh["negative electrode"][0].nodes, mesh["separator"][0].nodes]
            ),
            0,
        )
        with self.assertRaises(pybamm.DomainError):
            submesh = mesh.combine_submeshes("negative electrode", "positive electrode")

    def test_ghost_cells(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width [m]": 0.1,
                "Separator width [m]": 0.2,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
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
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"][0].edges[1],
            mesh["negative electrode"][0].edges[0],
        )
        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"][0].edges[0],
            -mesh["negative electrode"][0].edges[1],
        )
        np.testing.assert_array_equal(
            mesh["positive electrode_right ghost cell"][0].edges[0],
            mesh["positive electrode"][0].edges[-1],
        )

    def test_multiple_meshes(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width [m]": 0.1,
                "Separator width [m]": 0.2,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry("1+1D micro")
        param.process_geometry(geometry)

        # provide mesh properties

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_p: 10, var.r_n: 5, var.r_p: 6}
        submesh_types = {
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check types
        self.assertIsInstance(mesh["negative particle"], list)
        self.assertIsInstance(mesh["positive particle"], list)
        self.assertEqual(len(mesh["negative particle"]), 10)
        self.assertEqual(len(mesh["positive particle"]), 10)

        for i in range(10):
            self.assertIsInstance(mesh["negative particle"][i], pybamm.Uniform1DSubMesh)
            self.assertIsInstance(mesh["positive particle"][i], pybamm.Uniform1DSubMesh)
            self.assertEqual(mesh["negative particle"][i].npts, 5)
            self.assertEqual(mesh["positive particle"][i].npts, 6)

    def test_mesh_coord_sys(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width [m]": 0.1,
                "Separator width [m]": 0.2,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        for submeshlist in mesh.values():
            for submesh in submeshlist:
                self.assertTrue(submesh.coord_sys in pybamm.KNOWN_COORD_SYS)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
