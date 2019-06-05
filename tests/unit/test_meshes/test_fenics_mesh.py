#
# Test for the Finite Element (fenics) Mesh class
#
import pybamm
import unittest

import importlib

dolfin_spec = importlib.util.find_spec("dolfin")
if dolfin_spec is not None:
    dolfin = importlib.util.module_from_spec(dolfin_spec)
    dolfin_spec.loader.exec_module(dolfin)


@unittest.skipIf(dolfin_spec is None, "dolfin not installed")
class TestFenicsMesh(unittest.TestCase):
    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            base_parameters={
                "Electrode depth [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.1,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab width [m]": 0.1,
                "Positive tab centre y-coordinate [m]": 0.3,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode width [m]": 0.3,
                "Separator width [m]": 0.3,
                "Positive electrode width [m]": 0.3,
            }
        )

        geometry = pybamm.Geometryxp1DMacro(cc_dimension=2)
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.y: 16, var.z: 24}

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
        geometry = pybamm.Geometryxp1DMacro(cc_dimension=2)
        with self.assertRaises(KeyError):
            pybamm.Mesh(geometry, None, {})

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.y: 10, var.z: 10}
        with self.assertRaises(TypeError):
            pybamm.Mesh(geometry, None, var_pts)

        lims = {var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        with self.assertRaises(pybamm.DomainError):
            pybamm.FenicsMesh2D(lims, None, None)

        lims = {
            var.y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
            var.z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
        npts = {var.y.id: 10, var.z.id: 10}
        var.z.coord_sys = "not cartesian"
        with self.assertRaises(pybamm.DomainError):
            pybamm.FenicsMesh2D(lims, npts, None)
        var.z.coord_sys = "cartesian"

    def test_tab_placement(self):
        # set variables and submesh types
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 2, var.x_s: 2, var.x_p: 2, var.y: 64, var.z: 64}

        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.FenicsMesh2D,
        }

        mesh_type = pybamm.Mesh

        # set base parameters
        param = pybamm.ParameterValues(
            base_parameters={
                "Electrode depth [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab width [m]": 0.1,
                "Negative tab centre y-coordinate [m]": 0.1,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Negative electrode width [m]": 0.3,
                "Separator width [m]": 0.3,
                "Positive electrode width [m]": 0.3,
            }
        )

        # get negative tab location and size
        negative_tab_centre_y = param.process_symbol(
            pybamm.geometric_parameters.centre_y_tab_n
        ).evaluate()
        negative_tab_centre_z = param.process_symbol(
            pybamm.geometric_parameters.centre_z_tab_n
        ).evaluate()
        negative_tab_width = param.process_symbol(
            pybamm.geometric_parameters.l_tab_n
        ).evaluate()

        # loop over different positive tab placements and widths
        pos_tab_locations = {
            "top": {"y": 0.3, "z": 0.5, "width": 0.1},
            "bottom": {"y": 0.3, "z": 0, "width": 0.2},
            "left": {"y": 0, "z": 0.2, "width": 0.15},
            "right": {"y": 0.4, "z": 0.2, "width": 0.25},
        }

        for loc in pos_tab_locations.keys():
            param["Positive tab centre y-coordinate [m]"] = pos_tab_locations[loc]["y"]
            param["Positive tab centre z-coordinate [m]"] = pos_tab_locations[loc]["z"]
            param["Positive tab width [m]"] = pos_tab_locations[loc]["width"]

            # check geometry can be processed
            geometry = pybamm.Geometryxp1DMacro(cc_dimension=2)
            param.process_geometry(geometry)
            mesh = mesh_type(geometry, submesh_types, var_pts)

            # check tab location and size
            fenics_mesh = mesh["current collector"][0]
            positive_tab_centre_y = param.process_symbol(
                pybamm.geometric_parameters.centre_y_tab_p
            ).evaluate()
            positive_tab_centre_z = param.process_symbol(
                pybamm.geometric_parameters.centre_z_tab_p
            ).evaluate()
            positive_tab_width = param.process_symbol(
                pybamm.geometric_parameters.l_tab_p
            ).evaluate()
            self.assertAlmostEqual(
                fenics_mesh.negativetab.tab_location[0], negative_tab_centre_y
            )
            self.assertAlmostEqual(
                fenics_mesh.negativetab.tab_location[1], negative_tab_centre_z
            )
            self.assertAlmostEqual(
                fenics_mesh.positivetab.tab_location[0], positive_tab_centre_y
            )
            self.assertAlmostEqual(
                fenics_mesh.positivetab.tab_location[1], positive_tab_centre_z
            )
            # have to be lenient here with checking tab sizes
            self.assertAlmostEqual(
                dolfin.assemble(1 * fenics_mesh.ds(1)), negative_tab_width, places=1
            )
            self.assertAlmostEqual(
                dolfin.assemble(1 * fenics_mesh.ds(2)), positive_tab_width, places=1
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
