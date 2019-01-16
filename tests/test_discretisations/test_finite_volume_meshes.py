#
# Test for the Finite Volume Mesh class
#
import pybamm

import numpy as np
import unittest


class TestFiniteVolumeMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 50)
        self.assertEqual(mesh["whole cell"].edges[-1], 1)
        self.assertEqual(
            len(mesh["whole cell"].edges), len(mesh["whole cell"].nodes) + 1
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                mesh["whole cell"].nodes
                - np.concatenate(
                    [
                        mesh["negative electrode"].nodes,
                        mesh["separator"].nodes,
                        mesh["positive electrode"].nodes,
                    ]
                )
            ),
            0,
        )

    def test_mesh_sizes(self):
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.01, "Ls": 0.5, "Lp": 0.12}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 50)
        self.assertEqual(
            mesh.neg_mesh_points + (mesh.sep_mesh_points - 2) + mesh.pos_mesh_points,
            mesh.total_mesh_points,
        )
        self.assertEqual(mesh["negative electrode"].npts, mesh.neg_mesh_points - 1)
        self.assertEqual(mesh["separator"].npts, mesh.sep_mesh_points - 1)
        self.assertEqual(mesh["positive electrode"].npts, mesh.pos_mesh_points - 1)
        self.assertEqual(mesh["whole cell"].npts, mesh.total_mesh_points - 1)

        self.assertEqual(len(mesh["negative electrode"].edges), mesh.neg_mesh_points)
        self.assertEqual(len(mesh["separator"].edges), mesh.sep_mesh_points)
        self.assertEqual(len(mesh["positive electrode"].edges), mesh.pos_mesh_points)
        self.assertEqual(len(mesh["whole cell"].edges), mesh.total_mesh_points)

    def test_combine_submeshes(self):
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.01, "Ls": 0.5, "Lp": 0.12}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 50)
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
