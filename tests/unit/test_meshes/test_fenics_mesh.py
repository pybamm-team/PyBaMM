#
# Test for the Finite Element (fenics) Mesh class
#
import pybamm
import unittest


class TestMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Electrode depth [m]": 0.1,
                "Electrode height [m]": 0.2,
                "Negative tab width [m]": 0.01,
                "Negative tab centre y-coordinate [m]": 0.02,
                "Negative tab centre z-coordinate [m]": 0.2,
                "Positive tab width [m]": 0.01,
                "Positive tab centre y-coordinate [m]": 0.08,
                "Positive tab centre z-coordinate [m]": 0.2,
                "Negative electrode width [m]": 0.1,
                "Separator width [m]": 0.2,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry2p1DMacro()
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.y: 8, var.z: 10}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.FenicsMesh2D,
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
            if domain == "current collector":
                # NOTE: only for degree 1
                npts = var_pts[var.y] * var_pts[var.z]
                self.assertEqual(mesh[domain][0].npts, npts)
            else:
                self.assertEqual(
                    len(mesh[domain][0].edges), len(mesh[domain][0].nodes) + 1
                )

    def test_init_failure(self):
        geometry = pybamm.Geometry2p1DMacro()
        with self.assertRaises(KeyError):
            pybamm.Mesh(geometry, None, {})

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.y: 10, var.z: 10}
        geometry = pybamm.Geometry2p1DMacro()
        with self.assertRaises(TypeError):
            pybamm.Mesh(geometry, None, var_pts)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
