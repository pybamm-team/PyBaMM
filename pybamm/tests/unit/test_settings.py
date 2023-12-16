#
# Tests the settings class.
#
from tests import TestCase
import pybamm
import unittest


class TestSettings(TestCase):
    def test_simplify(self):
        self.assertTrue(pybamm.settings.simplify)

        pybamm.settings.simplify = False
        self.assertFalse(pybamm.settings.simplify)

        pybamm.settings.simplify = True

    def test_smoothing_parameters(self):
        self.assertEqual(pybamm.settings.min_smoothing, "exact")
        self.assertEqual(pybamm.settings.max_smoothing, "exact")
        self.assertEqual(pybamm.settings.heaviside_smoothing, "exact")
        self.assertEqual(pybamm.settings.abs_smoothing, "exact")

        pybamm.settings.set_smoothing_parameters(10)
        self.assertEqual(pybamm.settings.min_smoothing, 10)
        self.assertEqual(pybamm.settings.max_smoothing, 10)
        self.assertEqual(pybamm.settings.heaviside_smoothing, 10)
        self.assertEqual(pybamm.settings.abs_smoothing, 10)
        pybamm.settings.set_smoothing_parameters("exact")

        # Test errors
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            pybamm.settings.min_smoothing = -10
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            pybamm.settings.max_smoothing = -10
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            pybamm.settings.heaviside_smoothing = -10
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            pybamm.settings.abs_smoothing = -10


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
