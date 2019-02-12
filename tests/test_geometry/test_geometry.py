#
# Tests for the base model class
#
import pybamm


class StandardGeometryTests:
    def test_add_custom_geometry(self, geometry):

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.IndependentVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}
            }
        }

        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self, geometry):
        # checks that keys are pybamm.IndependentVariables
        keys = list(geometry.keys())
        all(self.assertIsInstance(key, pybamm.IndependentVariable) for key in keys)


class TestGeometry1DMacro:
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry1DMacro()
        tests = StandardGeometryTests()
        tests.test_add_custom_geometry(geometry)

    def test_geometry_keys(self):
        geometry = pybamm.Geometry1DMacro()
        tests = StandardGeometryTests(geometry)
        tests.test_add_custom_geometry(geometry)


class TestGeometry1DMicro:
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry1DMicro()
        tests = StandardGeometryTests()
        tests.test_add_custom_geometry(geometry)

    def test_geometry_keys(self):
        geometry = pybamm.Geometry1DMicro()
        tests = StandardGeometryTests(geometry)
        tests.test_add_custom_geometry(geometry)


class TestGeometry3DMacro:
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry3DMacro()
        tests = StandardGeometryTests()
        tests.test_add_custom_geometry(geometry)

    def test_geometry_keys(self):
        geometry = pybamm.Geometry3DMacro()
        tests = StandardGeometryTests(geometry)
        tests.test_add_custom_geometry(geometry)
