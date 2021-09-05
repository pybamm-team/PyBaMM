#
# Tests for the half-cell geometry class
#
import pybamm
import unittest
from pybamm.geometry import half_cell_spatial_vars
from pybamm.geometry.half_cell_geometry import half_cell_geometry


class TestHalfCellGeometry(unittest.TestCase):
    def test_geometry_keys(self):
        for working_electrode in ["positive", "negative"]:
            for cc_dimension in [0, 1, 2]:
                geometry = half_cell_geometry(
                    current_collector_dimension=cc_dimension,
                    working_electrode=working_electrode,
                )
                for domain_geoms in geometry.values():
                    all(
                        self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                        for spatial_var in domain_geoms.keys()
                    )

    def test_geometry(self):
        var = half_cell_spatial_vars
        geo = pybamm.geometric_parameters
        for working_electrode in ["positive", "negative"]:
            if working_electrode == "positive":
                l_w = geo.l_p
            else:
                l_w = geo.l_n
            for cc_dimension in [0, 1, 2]:
                geometry = half_cell_geometry(
                    current_collector_dimension=cc_dimension,
                    working_electrode=working_electrode,
                )
                self.assertIsInstance(geometry, pybamm.Geometry)
                self.assertIn("working electrode", geometry)
                self.assertIn("working particle", geometry)
                self.assertEqual(
                    geometry["working electrode"][var.x_w]["min"].id,
                    (geo.l_Li + geo.l_s).id,
                )
                self.assertEqual(
                    geometry["working electrode"][var.x_w]["max"].id,
                    (geo.l_Li + geo.l_s + l_w).id,
                )
                if cc_dimension == 1:
                    self.assertIn("tabs", geometry["current collector"])

        geometry = pybamm.battery_geometry(include_particles=False)
        self.assertNotIn("working particle", geometry)

    def test_geometry_error(self):
        with self.assertRaisesRegex(pybamm.GeometryError, "Invalid current"):
            half_cell_geometry(
                current_collector_dimension=4, working_electrode="positive"
            )
        with self.assertRaisesRegex(ValueError, "The option 'working_electrode'"):
            half_cell_geometry(working_electrode="bad electrode")


class TestReadParameters(unittest.TestCase):
    # This is the most complicated geometry and should test the parameters are
    # all returned for the deepest dict
    def test_read_parameters(self):
        geo = pybamm.geometric_parameters
        L_n = geo.L_n
        L_s = geo.L_s
        L_p = geo.L_p
        L_y = geo.L_y
        L_z = geo.L_z
        tab_n_y = geo.Centre_y_tab_n
        tab_n_z = geo.Centre_z_tab_n
        L_tab_n = geo.L_tab_n
        tab_p_y = geo.Centre_y_tab_p
        tab_p_z = geo.Centre_z_tab_p
        L_tab_p = geo.L_tab_p
        L_Li = geo.L_Li

        for working_electrode in ["positive", "negative"]:
            geometry = half_cell_geometry(
                current_collector_dimension=2, working_electrode=working_electrode
            )

            self.assertEqual(
                set([x.name for x in geometry.parameters]),
                set(
                    [
                        x.name
                        for x in [
                            L_n,
                            L_s,
                            L_p,
                            L_y,
                            L_z,
                            tab_n_y,
                            tab_n_z,
                            L_tab_n,
                            tab_p_y,
                            tab_p_z,
                            L_tab_p,
                            L_Li,
                        ]
                    ]
                ),
            )
            self.assertTrue(
                all(isinstance(x, pybamm.Parameter) for x in geometry.parameters)
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
