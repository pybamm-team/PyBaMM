#
# Tests for the base model class
#
import pybamm
import unittest


class TestGeometry1DMacro(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry1DMacro()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.IndependentVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}
            }
        }
        geometry.update(custom_geometry)
        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry1DMacro()
        for spatial_vars in geometry.values():
            all(
                self.assertIsInstance(spatial_var, pybamm.IndependentVariable)
                for spatial_var in spatial_vars.keys()
            )


class TestGeometry1DMicro(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry1DMicro()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.IndependentVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}
            }
        }
        geometry.update(custom_geometry)

        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry1DMicro()
        for spatial_vars in geometry.values():
            all(
                self.assertIsInstance(spatial_var, pybamm.IndependentVariable)
                for spatial_var in spatial_vars.keys()
            )


class TestGeometry3DMacro(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry3DMacro()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.IndependentVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}
            }
        }

        geometry.update(custom_geometry)
        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry3DMacro()
        for spatial_vars in geometry.values():
            all(
                self.assertIsInstance(spatial_var, pybamm.IndependentVariable)
                for spatial_var in spatial_vars.keys()
            )
