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
            self.assertEqual(len(mesh[domain].edges), len(mesh[domain].nodes) + 1)

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
            self.assertEqual(mesh[domain].npts, submesh_pts[domain]["x"])
            self.assertEqual(len(mesh[domain].edges) - 1, submesh_pts[domain]["x"])

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
        self.assertEqual(submesh.edges[0], 0)
        self.assertEqual(submesh.edges[-1], mesh["separator"].edges[-1])
        self.assertAlmostEqual(
            np.linalg.norm(
                submesh.nodes
                - np.concatenate(
                    [mesh["negative electrode"].nodes, mesh["separator"].nodes]
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
