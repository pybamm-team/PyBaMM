#
# Tests for the ParameterSets class
#
from tests import TestCase

import pybamm
import pkg_resources
import unittest


class TestParameterSets(TestCase):
    def test_name_interface(self):
        """Test that pybamm.parameters_sets.<ParameterSetName> returns
        the name of the parameter set and a depreciation warning
        """
        with self.assertWarns(DeprecationWarning):
            out = pybamm.parameter_sets.Marquis2019
            self.assertEqual(out, "Marquis2019")

        # Expect error for parameter set's that aren't real
        with self.assertRaises(AttributeError):
            pybamm.parameter_sets.not_a_real_parameter_set

    def test_all_registered(self):
        """Check that all parameter sets have been registered with the
        ``pybamm_parameter_sets`` entry point"""
        known_entry_points = set(
            ep.name for ep in pkg_resources.iter_entry_points("pybamm_parameter_sets")
        )
        self.assertEqual(set(pybamm.parameter_sets.keys()), known_entry_points)
        self.assertEqual(len(known_entry_points), len(pybamm.parameter_sets))

    def test_get_docstring(self):
        """Test that :meth:`pybamm.parameter_sets.get_doctstring` works"""
        docstring = pybamm.parameter_sets.get_docstring("Marquis2019")
        self.assertRegex(docstring, "Parameters for a Kokam SLPB78205130H cell")

    def test_iter(self):
        """Test that iterating `pybamm.parameter_sets` iterates over keys"""
        for k in pybamm.parameter_sets:
            self.assertIsInstance(k, str)
            self.assertIn(k, pybamm.parameter_sets)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
