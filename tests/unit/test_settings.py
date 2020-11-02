#
# Tests the settings class.
#
import pybamm
import unittest


class TestSettings(unittest.TestCase):
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

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            pybamm.settings.set_smoothing_parameters(-10)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
