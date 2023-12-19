#
# Tests for the Scalar class
#
from tests import TestCase
import unittest
import unittest.mock as mock

import pybamm


class TestScalar(TestCase):
    def test_scalar_eval(self):
        a = pybamm.Scalar(5)
        self.assertEqual(a.value, 5)
        self.assertEqual(a.evaluate(), 5)

    def test_scalar_operations(self):
        a = pybamm.Scalar(5)
        b = pybamm.Scalar(6)
        self.assertEqual((a + b).evaluate(), 11)
        self.assertEqual((a - b).evaluate(), -1)
        self.assertEqual((a * b).evaluate(), 30)
        self.assertEqual((a / b).evaluate(), 5 / 6)

    def test_scalar_eq(self):
        a1 = pybamm.Scalar(4)
        a2 = pybamm.Scalar(4)
        self.assertEqual(a1, a2)
        a3 = pybamm.Scalar(5)
        self.assertNotEqual(a1, a3)

    def test_to_equation(self):
        a = pybamm.Scalar(3)
        b = pybamm.Scalar(4)

        # Test value
        self.assertEqual(str(a.to_equation()), "3.0")

        # Test print_name
        b.print_name = "test"
        self.assertEqual(str(b.to_equation()), "test")

    def test_copy(self):
        a = pybamm.Scalar(5)
        b = a.create_copy()
        self.assertEqual(a, b)

    def test_to_from_json(self):
        a = pybamm.Scalar(5)
        json_dict = {"name": "5.0", "id": mock.ANY, "value": 5.0}

        self.assertEqual(a.to_json(), json_dict)

        self.assertEqual(pybamm.Scalar._from_json(json_dict), a)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
