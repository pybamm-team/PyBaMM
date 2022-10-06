#
# Tests for the ParameterSets class
#

import pybamm

import unittest


class TestParameterSets(unittest.TestCase):
    def test_parameter_sets(self):
        self.assertEqual(pybamm.parameter_sets.Marquis2019, "Marquis2019")
        with self.assertRaises(AttributeError):
            pybamm.parameter_sets.not_a_real_parameter_set


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
