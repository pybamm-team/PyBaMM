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
                "Negative electrode width": 0.1,
                "Separator width": 0.2,
                "Positive electrode width": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        # provide mesh properties
        submesh_pts = {
            "negative electrode": {"x": 10},
            "separator": {"x": 10},
            "positive electrode": {"x": 12},
            "negative particle": {"r": 5},
            "positive particle": {"r": 6},
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, submesh_pts)

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

    def test_mesh_sizes(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width": 0.1,
                "Separator width": 0.2,
                "Positive electrode width": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        # provide mesh properties
        submesh_pts = {
            "negative electrode": {"x": 10},
            "separator": {"x": 10},
            "positive electrode": {"x": 12},
            "negative particle": {"r": 5},
            "positive particle": {"r": 6},
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, submesh_pts)
        for domain in mesh:
            self.assertEqual(mesh[domain][0].npts, submesh_pts[domain]["x"])
            self.assertEqual(len(mesh[domain][0].edges) - 1, submesh_pts[domain]["x"])

    def test_combine_submeshes(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width": 0.1,
                "Separator width": 0.2,
                "Positive electrode width": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        # provide mesh properties
        submesh_pts = {
            "negative electrode": {"x": 10},
            "separator": {"x": 10},
            "positive electrode": {"x": 12},
            "negative particle": {"r": 5},
            "positive particle": {"r": 6},
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, submesh_pts)

        # create submesh
        submesh = mesh.combine_submeshes("negative electrode", "separator")
        self.assertEqual(submesh[0].edges[0], 0)
        self.assertEqual(submesh[0].edges[-1], mesh["separator"][0].edges[-1])
        self.assertAlmostEqual(
            np.linalg.norm(
                submesh[0].nodes
                - np.concatenate(
                    [mesh["negative electrode"][0].nodes, mesh["separator"][0].nodes]
                )
            ),
            0,
        )
        with self.assertRaises(pybamm.DomainError):
            submesh = mesh.combine_submeshes("negative electrode", "positive electrode")

    def test_ghost_cells(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width": 0.1,
                "Separator width": 0.2,
                "Positive electrode width": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        # provide mesh properties
        submesh_pts = {
            "negative electrode": {"x": 10},
            "separator": {"x": 10},
            "positive electrode": {"x": 12},
            "negative particle": {"r": 5},
            "positive particle": {"r": 6},
        }
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, submesh_pts)

        mesh.add_ghost_meshes()

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
                "Negative electrode width": 0.1,
                "Separator width": 0.2,
                "Positive electrode width": 0.3,
            }
        )

        geometry = pybamm.Geometry("1+1D micro")
        param.process_geometry(geometry)

        # provide mesh properties
        submesh_pts = {
            "negative particle": {"r": 5, "x": 10},
            "positive particle": {"r": 6, "x": 11},
        }
        submesh_types = {
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

        mesh = pybamm.Mesh(geometry, submesh_types, submesh_pts)

        # check types
        self.assertIsInstance(mesh["negative particle"], list)
        self.assertIsInstance(mesh["positive particle"], list)
        self.assertEqual(len(mesh["negative particle"]), 10)
        self.assertEqual(len(mesh["positive particle"]), 11)

        for i in range(1):
            self.assertIsInstance(mesh["negative particle"][i], pybamm.Uniform1DSubMesh)
            self.assertIsInstance(mesh["positive particle"][i], pybamm.Uniform1DSubMesh)
            self.assertEqual(mesh["negative particle"][i].npts, 5)
            self.assertEqual(mesh["positive particle"][i].npts, 6)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
