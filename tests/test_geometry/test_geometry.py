#
# Tests for the base model class
#
import pybamm


class TestGeometry:
    def test_add_custom_geometry(self):
        custom_geometry = {
            "negative electrode": {
                "x": {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}
            }
        }
        geometry = pybamm.Geometry1DMacro(custom_geometry)

        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    # process geometry test conducted within TestParameterValues
